import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler
import torch_dct as dct
from eval import EvaluatorModelWrapper
from options.evaluate_options import TestOptions
import time
from os.path import join as pjoin
import torch.optim as optim

class DDPMTrainer(object):

    def __init__(self, args, model, accelerator, model_ema=None):
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.model = model
        self.diffusion_steps = args.diffusion_steps
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule=args.beta_schedule,
            variance_type="fixed_small",
            prediction_type=args.prediction_type,
            clip_sample=False
        )
        self.model_ema = model_ema
        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.lf_loss = torch.nn.MSELoss()

        accelerator.print('Diffusion_config:\n', self.noise_scheduler.config)

        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            self.writer = SummaryWriter(
                log_dir=pjoin(args.save_root, 'logs_') + starttime[:16],
                comment=starttime[:16],
                flush_secs=60
            )
        self.accelerator.wait_for_everyone()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.opt.lr,
            weight_decay=self.opt.weight_decay
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate > 0 else None
        parser = TestOptions()
        opt = {
            'dataset_name': 'humanml',
            'device': self.device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': 15,
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263,  # 263 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '/home/user/dxc/motion/StableMoFusion/data/pretrained_models/',
            'unit_length': 4,
        }
        self.eval_wrapper = EvaluatorModelWrapper(opt)
        self.num = 0

        # Define the threshold for early and late steps (e.g., 30% for early steps)
        self.lf_threshold = int(0.3 * self.diffusion_steps)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self, network_list):
        for network in network_list:
            self.accelerator.clip_grad_norm_(network.parameters(), self.opt.clip_grad_norm)  # 0.5 -> 1

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        caption, motions, m_lens = batch_data
        motions = motions.detach().float()

        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(self.device)
        self.src_mask = self.generate_src_mask(T, cur_len).to(x_start.device)

        # 1. Sample noise that we'll add to the motion
        real_noise = torch.randn_like(x_start)

        # 2. Sample a random timestep for each motion
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t

        # 3. Add noise to the motion according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)

        # 4. network prediction
        self.prediction = self.model(x_t, t, text=caption)

        if self.opt.prediction_type == 'sample':
            self.target = x_start
        elif self.opt.prediction_type == 'epsilon':
            self.target = real_noise
        elif self.opt.prediction_type == 'v_prediction':
            self.target = self.noise_scheduler.get_velocity(x_start, real_noise, t)

    def masked_l2(self, a, b, mask, weights):
        # Base MSE loss
        loss = self.mse_criterion(a, b).mean(dim=-1)  # (batch_size, motion_length)
        loss = (loss * mask).sum(-1) / mask.sum(-1)  # (batch_size, )
        loss = (loss * weights).mean()

        # Initialize additional losses
        loss_lf = torch.tensor(0.0, device=self.device)
        loss_s = torch.tensor(0.0, device=self.device)

        # Apply loss_lf for early steps
        lf_mask = self.timesteps <= self.lf_threshold
        if lf_mask.any():
            gt_lf = self.remove_lf(a[lf_mask].squeeze())
            model_output__lf = self.remove_lf(b[lf_mask].squeeze())
            loss_lf = self.lf_loss(gt_lf, model_output__lf)
            loss += loss_lf

        # Apply loss_s for late steps
        s_mask = self.timesteps > self.lf_threshold
        if s_mask.any():
            mlens = []
            for i in b[s_mask]:
                n = 0
                for j in i:
                    if j.sum() != 0:
                        n += 1
                mlens.append(n)

            motion_embedding = self.eval_wrapper.get_motion_embeddings_train(a[s_mask], m_lens=torch.tensor(mlens).to(self.device))
            mlens = []
            for i in b[s_mask]:
                n = 0
                for j in i:
                    if j.sum() != 0:
                        n += 1
                mlens.append(n)

            target_motion_embedding = self.eval_wrapper.get_motion_embeddings_train(b[s_mask], m_lens=torch.tensor(mlens).to(self.device))

            cosine_similarity = F.cosine_similarity(target_motion_embedding, motion_embedding.to(target_motion_embedding), dim=1)
            loss_s = 1 - cosine_similarity.mean()
            loss += loss_s

        # Optional: Print the loss components for debugging
        if self.num > 10:
            print(f"Total Loss: {loss.item()} | loss_lf: {loss_lf.item()} | loss_s: {loss_s.item()}")

        self.num += 1

        return loss

    def dct_idct(self, vec, bs, dim=1):
        vec_f = dct.dct(vec.reshape(bs, -1))
        vec_f[:, dim:] = 0
        return dct.idct(vec_f)

    def remove_lf(self, batch):
        bs = batch.shape[0]
        vecs_list = [self.dct_idct(vec.reshape(bs, -1), bs).unsqueeze(-1) for vec in batch.split(1, dim=-1)]
        return torch.cat(vecs_list, dim=-1).reshape(bs, -1, 263)

    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        loss_logs['loss_mot_rec'] = self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)

        self.loss = loss_logs['loss_mot_rec']

        return loss_logs

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        self.clip_norm([self.model])
        self.step([self.optimizer])

        return loss_logs

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T, device=self.device)
        for i in range(B):
            if length[i] < T:
                src_mask[i, length[i]:] = 0
        return src_mask

    def train_mode(self):
        self.model.train()
        if self.model_ema:
            self.model_ema.train()

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()

    def save(self, file_name, total_it):
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.model_ema:
            state["model_ema"] = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
        torch.save(state, file_name)

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        print(f"Loaded model from {model_dir}")
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint.get('total_it', 0)

    def train(self, train_loader):
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)
            it = self.load(model_path)
            self.accelerator.print(f'Continue training from {it} iterations in {model_path}')
        start_time = time.time()

        logs = OrderedDict()
        self.dataset = train_loader.dataset
        self.model, self.mse_criterion, self.optimizer, train_loader, self.model_ema = \
            self.accelerator.prepare(self.model, self.mse_criterion, self.optimizer, train_loader, self.model_ema)

        num_epochs = (self.opt.num_train_steps - it) // len(train_loader) + 1
        self.accelerator.print(f'Need to train for {num_epochs} epochs...')

        for epoch in range(num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                it += 1

                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(self.model_ema).update_parameters(self.model)

                # Update logger
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(self.accelerator, start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("loss", mean_loss['loss_mot_rec'], it)
                    self.accelerator.wait_for_everyone()

                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)
                self.accelerator.wait_for_everyone()

                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0):
                    self.scheduler.step()

                if it >= self.opt.num_train_steps:
                    break

            if it >= self.opt.num_train_steps:
                break

        # Save the last checkpoint if it wasn't already saved.
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISHED TRAINING')
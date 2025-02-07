# This file code from T2M(https://github.com/EricGuo5513/text-to-motion), licensed under the https://github.com/EricGuo5513/text-to-motion/blob/main/LICENSE.
# Copyright (c) 2022 Chuan Guo
from datetime import datetime
import numpy as np
import torch
from utils.metrics import *
from collections import OrderedDict

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluate_matching_score(eval_wrapper,motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()
                # import pdb;pdb.set_trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            # import pdb;pdb.set_trace()
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper,groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


# def get_metric_statistics(values, replication_times):
#     mean = np.mean(values, axis=0)
#     std = np.std(values, axis=0)
#     conf_interval = 1.96 * std / np.sqrt(replication_times)
#     return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    with open(log_file, 'a') as f:
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict(),
            'R_precision': OrderedDict(),
            'FID': OrderedDict(),
            'Diversity': OrderedDict(),
            'MultiModality': OrderedDict()
        })

        for replication in range(replication_times):
            print(f'\nTime: {datetime.now()}')
            print(f'\nTime: {datetime.now()}', file=f, flush=True)
            
            # Initialize loaders
            motion_loaders = {'ground truth': gt_loader}
            mm_motion_loaders = {}
            
            # Load generated motions
            for name, getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader, gen_time = getter()
                motion_loaders[name] = motion_loader
                mm_motion_loaders[name] = mm_motion_loader
                print(f'---> [{name}] Generation time: {gen_time:.2f}s', file=f, flush=True)

            # Single replication evaluation
            if replication_times > 1:
                print(f'\n==================== Replication {replication+1}/{replication_times} ====================')
                print(f'\n==================== Replication {replication+1}/{replication_times} ====================', 
                      file=f, flush=True)

            # Main evaluation steps
            mat_score_dict, R_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)
            fid_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)
            div_dict = evaluate_diversity(acti_dict, f, diversity_times)
            mm_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times) if run_mm else {}

            # Collect metrics
            for metric_dict, metric_name in zip(
                [mat_score_dict, R_dict, fid_dict, div_dict, mm_dict],
                ['Matching Score', 'R_precision', 'FID', 'Diversity', 'MultiModality']
            ):
                for name, value in metric_dict.items():
                    if name not in all_metrics[metric_name]:
                        all_metrics[metric_name][name] = []
                    all_metrics[metric_name][name].append(value)

        # Final statistics
        final_metrics = OrderedDict()
        if replication_times > 1:
            print('\n\n==================== Final Results ====================')
            print('\n==================== Final Results ====================', file=f, flush=True)
            
            for metric_name, metric_dict in all_metrics.items():
                print(f'\n========== {metric_name} Summary ==========')
                print(f'\n========== {metric_name} Summary ==========', file=f, flush=True)
                
                for model_name, values in metric_dict.items():
                    values = np.array(values)
                    mean = np.mean(values, axis=0)
                    std = np.std(values, axis=0)
                    conf_interval = 1.96 * std / np.sqrt(replication_times)
                    
                    # Format output with ±
                    if values.ndim == 1:  # Single value metrics
                        final_metrics[f'{metric_name}_{model_name}'] = mean
                        line = (f'---> [{model_name}] '
                                f'{mean[0]:.4f} ± {conf_interval[0]:.4f}' 
                                if values[0].ndim == 0 
                                else f'{mean[0]:.4f} ± {conf_interval[0]:.4f}')
                    else:  # Multi-value metrics (e.g. R_precision)
                        final_metrics[f'{metric_name}_{model_name}'] = mean
                        line = '---> [{}] '.format(model_name) + ' '.join(
                            [f'(top {i+1}): {m:.4f} ± {c:.4f}'
                             for i, (m, c) in enumerate(zip(mean, conf_interval))]
                        )
                    
                    print(line)
                    print(line, file=f, flush=True)

        else:  # Single replication
            final_metrics = all_metrics
            for metric_name, metric_dict in all_metrics.items():
                print(f'\n========== {metric_name} ==========')
                print(f'\n========== {metric_name} ==========', file=f, flush=True)
                for name, values in metric_dict.items():
                    print(f'---> [{name}]: {values[0]}')
                    print(f'---> [{name}]: {values[0]}', file=f, flush=True)

        return final_metrics
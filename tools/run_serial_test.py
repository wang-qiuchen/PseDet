import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='train_serial')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--ckpt', help='the dir to save logs and models')
    parser.add_argument('--port',default='29505', help='the dir to save logs and models')
    args = parser.parse_args()
    portcfg = 'env_cfg.dist_cfg.port='+args.port
    ckpt = args.ckpt
    config0 = args.config
    config1 = config0.replace('D0','D1')
    config2 = config0.replace('D0','D2')

    scripts=[
        ['tools/test.py', config0, ckpt,'--launcher=slurm','--cfg-options',portcfg],
        ['tools/test.py', config1, ckpt,'--launcher=slurm','--cfg-options',portcfg],
        ['tools/test.py', config2, ckpt,'--launcher=slurm','--cfg-options',portcfg]
    ]
    print(scripts)
    # # 串行执行脚本
    for script in scripts:
        subprocess.run(['python', *script], check=True)

if __name__ == '__main__':
    main()

# # 定义要调用的脚本文件和参数
# scripts = [
#     ['tools/test.py', 'bdd_cfg/sdmv3_learnable_init_proj_and_mlp_former/bdd_atss_r50_1x_sdm_version3_seq0123_D0_learnable_init_proj_and_mlp_former.py', 'work_dirs/bdd_atss_r50_1x_sdm_version3_seq0123_D3_learnable_init_proj_and_mlp_former/epoch_12.pth',
#     '--launcher=slurm','--cfg-options','env_cfg.dist_cfg.port=29505'],
#     ['tools/test.py', 'bdd_cfg/sdmv3_learnable_init_proj_and_mlp_former/bdd_atss_r50_1x_sdm_version3_seq0123_D1_learnable_init_proj_and_mlp_former.py', 'work_dirs/bdd_atss_r50_1x_sdm_version3_seq0123_D3_learnable_init_proj_and_mlp_former/epoch_12.pth',
#     '--launcher=slurm','--cfg-options','env_cfg.dist_cfg.port=29505'],
#     ['tools/test.py', 'bdd_cfg/sdmv3_learnable_init_proj_and_mlp_former/bdd_atss_r50_1x_sdm_version3_seq0123_D2_learnable_init_proj_and_mlp_former.py', 'work_dirs/bdd_atss_r50_1x_sdm_version3_seq0123_D3_learnable_init_proj_and_mlp_former/epoch_12.pth',
#     '--launcher=slurm','--cfg-options','env_cfg.dist_cfg.port=29505']
# ]
# # 串行执行脚本
# for script in scripts:
#     subprocess.run(['python', *script], check=True)

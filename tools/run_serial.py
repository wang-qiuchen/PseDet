import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='train_serial')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--port',default='29505', help='the dir to save logs and models')
    parser.add_argument('--ckpt', help='the dir to save logs and models')

    args = parser.parse_args()
    ckpt = args.ckpt
    portcfg = 'env_cfg.dist_cfg.port='+args.port
    config0 = args.config
    workdir0 = '--work-dir='+args.work_dir
    config1 = config0.replace('D0','D1')
    workdir1 = workdir0.replace('D0','D1')
    config2 = config0.replace('D0','D2')
    workdir2 = workdir0.replace('D0','D2')
    config3 = config0.replace('D0','D3')
    workdir3 = workdir0.replace('D0','D3')
    scripts=[
        ['tools/train.py', config0, workdir0, '--launcher=slurm', '--cfg-options', portcfg, '--resume'],
        ['tools/train.py', config1, workdir1, '--launcher=slurm', '--cfg-options', portcfg, '--resume'],
        ['tools/train.py', config2, workdir2, '--launcher=slurm', '--cfg-options', portcfg, '--resume'],
        ['tools/train.py', config3, workdir3, '--launcher=slurm', '--cfg-options', portcfg, '--resume']
    ]
    if ckpt is not None:
        scripts.extend(
            [['tools/test.py', config0, ckpt,'--launcher=slurm','--cfg-options',portcfg],
            ['tools/test.py', config1, ckpt,'--launcher=slurm','--cfg-options',portcfg],
            ['tools/test.py', config2, ckpt,'--launcher=slurm','--cfg-options',portcfg]]
        )
    print(scripts)
    # # 串行执行脚本
    for script in scripts:
        subprocess.run(['python', *script], check=True)

if __name__ == '__main__':
    main()
import click
import os

ROOT_PATH = os.path.join('data', 'results', 'rfidsim')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config_name')
@click.option('--max-real-time', type=float, 
              help="If given, simulation will stop after this time.")
@click.option('-s', '--set-value', 'values', type=str, multiple=True,
              help="Override configuration parameters in the form "\
                   "'arg:value', e.g. vehicle.speed:60")
def simulate(config_name, max_real_time, values):
    os.makedirs(ROOT_PATH, exist_ok=True)
    info_file_name = f'rfidsim_{config_name}_info.json'
    with open(os.path.join(ROOT_PATH, info_file_name), 'w') as f:
        f.writelines(['[]'])

"""
Main entry point for the panobinostat ridge reproduction and LIME analysis.
Loads pre-packaged EC11K/MC9K artefacts, evaluates ridge performance under
provided splits and repeated stratified CV, and generates JSON, figures, and
a LaTeX/PDF report.
"""

from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from compare_and_lime import compare_panobinostat, lime_panobinostat
from create import (
    render_ridge_section,
    render_report_tex,
    build_pdf_from_tex,
    write_figures,
)

# Configuration controlling evaluation, LIME settings, and report generation.

REPORT_CONFIG = {
    'build_pdf': False,
    'lime_num_cases': 4,
    'lime_num_features': 10,
    'lime_label': 'EC11K',
    'lime_split': 1,
    'seed': 0,
    'cv_n_splits': 5,
    'cv_n_repeats': 10,
    'cv_y_bins': 10,
}

def _to_jsonable(x):
    """
    Converts NumPy scalars and arrays to native Python types for JSON use
    """
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def load_data(root):
    """
    Load EC11K and MC9K panobinostat data and predefined train-test splits
    """
    data = {}
    for label in ('EC11K', 'MC9K'):
        z = np.load(root / 'data' / f'{label}_Panobinostat.npz')
        X, y = z['x'], z['y']
        splits = {}
        for i in range(1, 4):
            s = np.load(root / 'data' / f'{label}_Panobinostat_{i}.npz')
            splits[i] = {
                'X_train': X[s['train']], 
                'y_train': y[s['train']], 
                'X_test': X[s['test']], 
                'y_test': y[s['test']],
            }
        data[label] = {'X': X, 'y': y, 'splits': splits}
    return data

def run_analysis(config):
    """
    Run ridge reproduction (provided splits and repeated stratified CV) and LIME analysis for panobinostat, 
    returning all results needed for reporting. 
    """
    root = Path(__file__).resolve().parent
    model = Ridge(alpha=0.001, solver='svd')
    data = load_data(root)
    seed = int(config.get('seed', 0))
    cv_kwargs = {
        'cv_n_splits': int(config.get('cv_n_splits', 5)),
        'cv_n_repeats': int(config.get('cv_n_repeats', 10)),
        'cv_y_bins': int(config.get('cv_y_bins', 10)),
    }
    ridge_sections = []
    print('Running provided splits...', flush=True)
    rows_p, summaries_p, delta_p = compare_panobinostat(
        data,
        model,
        mode='provided',
        seed=seed,
        **cv_kwargs,
    )
    print('Provided splits done.', flush=True)
    ridge_sections.append(render_ridge_section(rows_p, summaries_p, delta_p, 'provided'))
    print('Running CV...', flush=True)
    rows_c, summaries_c, delta_c = compare_panobinostat(
        data,
        model,
        mode='cv',
        seed=seed,
        **cv_kwargs,
    )
    print('CV done.', flush=True)
    ridge_sections.append(render_ridge_section(
        rows_c, summaries_c, delta_c, 'cv',
        cv_n_splits=cv_kwargs['cv_n_splits'],
        cv_n_repeats=cv_kwargs['cv_n_repeats'],
    ))
    print('Running LIME...', flush=True)
    lime_data = lime_panobinostat(
        data,
        model,
        num_cases=int(config.get('lime_num_cases', 4)),
        num_features=int(config.get('lime_num_features', 10)),
        label=str(config.get('lime_label', 'EC11K')),
        split=int(config.get('lime_split', 1)),
        seed=seed,
    )
    print('LIME done.', flush=True)
    results = {
        'seed': seed,
        'config': dict(config),
        'provided': {
             'rows': rows_p,
             'summaries': summaries_p,
             'delta_r2': delta_p,
        },
        'cv': {
             'rows': rows_c,
             'summaries': summaries_c,
             'delta_r2': delta_c,
        },
        'lime_data': lime_data,
        'data': data,
        'model': model,
        'ridge_sections': ridge_sections,
    }
    return results

def main():
    """
    Execute the full panobinostat analysis pipeline, writing JSON results, 
    figures, and a LaTeX report (optionally compiled to PDF) to the outputs directory.
    """
    report_title = 'Panobinostat: Ridge reproduction and LIME'
    root = Path(__file__).resolve().parent
    outdir = root / 'outputs'
    outdir.mkdir(exist_ok=True, parents=True)
    tex_path = outdir / 'report.tex'
    results = run_analysis(REPORT_CONFIG)
    json_path = outdir / 'results.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(
            _to_jsonable({
                'seed': results['seed'],
                'config': results['config'],
                'provided': results['provided'],
                'cv': results['cv'],
                'lime_data': results['lime_data'],
            }),
            f,
            indent=2,
            sort_keys=True,
        )
    data = results['data']
    model = results['model']
    seed = results['seed']
    ridge_sections = results['ridge_sections']
    rows_cv = results['cv']['rows']
    lime_data = results['lime_data']
    fig_cv_name, fig_scatter_name = write_figures(
        outdir=outdir,
        data=data,
        model=model,
        rows_cv=rows_cv,
        seed=seed,
    )
    tex = render_report_tex(
        lime_data=lime_data, 
        ridge_sections=ridge_sections,
        fig_cv_name=fig_cv_name,
        fig_scatter_name=fig_scatter_name,
        report_title=report_title,
    )
    tex_path.write_text(tex, encoding='utf-8')
    if REPORT_CONFIG.get('build_pdf', False):
        try:
            build_pdf_from_tex(tex_path)
        except FileNotFoundError:
            raise RuntimeError(
                "pdflatex not found. Either install a LaTeX distribution (e.g. TeX Live) "
                "or set REPORT_CONFIG['build_pdf'] = False."
            )
        for ext in ('.aux', '.log', '.out'):
            p = tex_path.with_suffix(ext)
            if p.exists():
                p.unlink()

if __name__ == '__main__':
    main()
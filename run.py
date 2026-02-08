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
    build_report_title, 
)

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
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def load_data(root):
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
    rows_p, summaries_p, delta_p = compare_panobinostat(
        data,
        model,
        mode='provided',
        seed=seed,
        **cv_kwargs,
    )
    ridge_sections.append(render_ridge_section(rows_p, summaries_p, delta_p, 'provided'))
    rows_c, summaries_c, delta_c = compare_panobinostat(
        data,
        model,
        mode='cv',
        seed=seed,
        **cv_kwargs,
    )
    rows_cv = rows_c
    ridge_sections.append(render_ridge_section(
        rows_c, summaries_c, delta_c, 'cv',
        cv_n_splits=int(config.get('cv_n_splits', 5)),
        cv_n_repeats=int(config.get('cv_n_repeats', 10)),
    ))
    lime_data = lime_panobinostat(
        data,
        model,
        num_cases=int(config.get('lime_num_cases', 4)),
        num_features=int(config.get('lime_num_features', 10)),
        label=str(config.get('lime_label', 'EC11K')),
        split=int(config.get('lime_split', 1)),
        seed=seed,
    )
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
        'rows_cv': rows_cv,
    }
    return results

def main():
    report_title = build_report_title(REPORT_CONFIG)
    root = Path(__file__).resolve().parent
    outdir = root / 'outputs'
    outdir.mkdir(exist_ok=True)
    tex_path = outdir / 'report.tex'
    results = run_analysis(REPORT_CONFIG)
    json_path = outdir / 'results.json'
    with json_path.open('w', encoding='utf-8') as f:
         json.dump(_to_jsonable({
              'seed': results['seed'], 
              'config': results['config'], 
              'provided': results['provided'], 
              'cv': results['cv'], 
              'lime_data': results['lime_data']
              }),
              f,
              indent=2,
              sort_keys=True,
        )
    data = results['data']
    model = results['model']
    seed = results['seed']
    ridge_sections = results['ridge_sections']
    rows_cv = results['rows_cv']
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
    if bool(REPORT_CONFIG.get('build_pdf', True)):
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
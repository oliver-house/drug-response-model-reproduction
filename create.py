"""
Reporting and output utilities for the panobinostat reproduction study.
Provides LaTeX rendering, figure generation, and optional PDF compilation
from computed ridge and LIME results.
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

def feature_to_tex(s):
    """
    Escape special characters and comparison operators in a string for safe inclusion in LaTeX.
    """
    s = str(s).replace('_', r'\_')
    s = s.replace('<=', r'\(\le\)')
    s = s.replace('>=', r'\(\ge\)')
    s = s.replace('<', r'\(<\)')
    s = s.replace('>', r'\(>\)')
    s = s.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
    return s

def render_ridge_section(
    rows, 
    summaries, 
    delta_r2, 
    compare_mode, 
    cv_n_splits=None, 
    cv_n_repeats=None,
    ):
    """
    Render a LaTeX section summarising ridge performance for EC11K and MC9K, including per-split (or CV) results, 
    summary statistics, and the mean R² difference.
    """
    ridge_section = ''
    if rows is not None and summaries is not None and delta_r2 is not None:
        if str(compare_mode) == 'provided':
            eval_note = '\\noindent Evaluation uses the 3 provided train-test splits.'
            show_ridge_table = True
        elif str(compare_mode) == 'cv':
            n_splits = int(cv_n_splits) if cv_n_splits is not None else 0
            n_repeats = int(cv_n_repeats) if cv_n_repeats is not None else 0
            n_total = n_splits * n_repeats if (n_splits > 0 and n_repeats > 0) else len(rows)
            eval_note = (
                f'\\noindent Evaluation uses stratified {int(n_splits)}-fold '
                f'cross-validation repeated {int(n_repeats)} times ({int(n_total)} folds).'
            )
            show_ridge_table = False
        else:
            eval_note = f'\\noindent Evaluation mode: {feature_to_tex(compare_mode)}.'
            show_ridge_table = True
        ridge_table_body = (
            '\n'.join(
                f"{feature_to_tex(r['dataset'])} & {int(r['split'])} & "
                f"{float(r['R2']):.3f} & {float(r['RMSE']):.3f} \\\\"
                for r in rows
            )
            if show_ridge_table
            else ''
        )
        ridge_summary_table_body = '\n'.join((
            (
                f"EC11K & ${summaries['EC11K']['R2_mean']:.3f} "
                f"\\pm {summaries['EC11K']['R2_sd']:.3f}$ & "
                f"${summaries['EC11K']['RMSE_mean']:.3f} "
                f"\\pm {summaries['EC11K']['RMSE_sd']:.3f}$ \\\\"
            ),
            (
                f"MC9K & ${summaries['MC9K']['R2_mean']:.3f} "
                f"\\pm {summaries['MC9K']['R2_sd']:.3f}$ & "
                f"${summaries['MC9K']['RMSE_mean']:.3f} "
                f"\\pm {summaries['MC9K']['RMSE_sd']:.3f}$ \\\\"
            ),
        ))
        ridge_table_tex = (
            '\n'.join((
                '\\begin{center}',
                '\\begin{tabular}{lrrr}',
                '\\toprule',
                'Dataset & Split & $R^2$ & RMSE \\\\',
                '\\midrule',
                f'{ridge_table_body}',
                '\\bottomrule',
                '\\end{tabular}',
                '\\end{center}',
            ))
            if show_ridge_table
            else ''
        )
        ridge_section = '\n'.join((
            (
                '\\section{{Ridge reproduction (provided splits): EC11K vs MC9K}}'
                if str(compare_mode) == 'provided'
                else '\\section{{Ridge reproduction (repeated stratified CV): EC11K vs MC9K}}'
                if str(compare_mode) == 'cv'
                else f'\\section{{Ridge reproduction ({feature_to_tex(compare_mode)}): EC11K vs MC9K}}'
            ),
            f'{eval_note}',
            f'{ridge_table_tex}',
            '\\subsection*{{Summary across folds}}' if str(compare_mode) == 'cv' else '\\subsection*{{Summary across splits}}',
            '\\begin{center}',
            '\\begin{tabular}{lrr}',
            '\\toprule',
            'Dataset & $R^2$ (mean $\\pm$ SD) & RMSE (mean $\\pm$ SD) \\\\',
            '\\midrule',
            f'{ridge_summary_table_body}',
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{center}',
            f'$\\Delta R^2$ (EC11K$-$MC9K) = {float(delta_r2):.3f}.'
        ))
    return ridge_section

def render_report_tex(
        lime_data=None, 
        ridge_sections=None,
        fig_cv_name=None,
        fig_scatter_name=None,
        report_title='Panobinostat report',
):  
    """
    Assemble and return the complete LaTeX document for the report, combining ridge results, figures, 
    and LIME explanations into a single article template.
    """
    ridge_section = '\n\n'.join([s for s in ridge_sections if str(s).strip() != ''])
    case_summary_table_body = '\n'.join(
        f"{c['case']} & {int(c['test_row'])} & "
        f"{float(c['pred_lnIC50']):.3f} & {float(c['obs_lnIC50']):.3f} & "
        f"{(float(c['pred_lnIC50']) - float(c['obs_lnIC50'])):+.3f} \\\\"
        for c in lime_data['cases']
    )
    lime_case_tables = []
    for c in lime_data['cases']:
        case_id = int(c['case'])
        test_row = int(c['test_row'])
        pred = float(c['pred_lnIC50'])
        obs = float(c['obs_lnIC50'])
        feature_rows = []
        for rank, feat in enumerate(c['features'], start=1):
            feature_rows.append(
                f"{rank} & {feature_to_tex(feat['feature'])} & "
                f"{float(feat['weight']):.4f} \\\\"
            )
        feature_table_body = '\n'.join(feature_rows)
        lime_case_tables.append('\n'.join((
            (
                f'\\subsubsection{{Case {case_id} (test row {test_row}; '
                f'$\\widehat{{\\ln(\\mathrm{{IC50}})}}={pred:.3f}$, '
                f'$\\ln(\\mathrm{{IC50}})={obs:.3f}$)}}'
            ),
            '\\begin{center}',
            '\\begin{tabular}{r p{110mm} r}',
            '\\toprule',
            'Rank & Feature (local LIME rule) & Weight \\\\',
            '\\midrule',
            f'{feature_table_body}',
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{center}',
        )))
    lime_case_tables_tex = '\n'.join(lime_case_tables)
    lime_section = '\n'.join((
        '\\section{{LIME: local explanations (top 4 most sensitive predictions)}}',
        (
            '\\noindent Cases correspond to the 4 smallest predicted values of '
            '$\\ln(\\mathrm{{IC50}})$ in the selected test set.'
        ),
        '\\subsection{{Case summaries}}',
        '\\begin{center}',
        '\\begin{tabular}{rrrrr}',
        '\\toprule',
        (
            'Case & Test row & $\\widehat{{\\ln(\\mathrm{{IC50}})}}$ '
            '& $\\ln(\\mathrm{{IC50}})$ & Residual \\\\'
        ),
        '\\midrule',
        f'{case_summary_table_body}',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{center}',
        '\\subsection{{Top features per case (weights)}}',
        (
            '\\emph{{Weights are local linear surrogate coefficients from LIME; '
            'magnitude reflects local influence, sign indicates direction.}}'
        ),
        f'{lime_case_tables_tex}',
    ))
    fig_lines = []
    fig_lines.append('\n'.join((
        '\\section{Figures}',
        '\\subsection{Cross-validation $R^2$ distribution}',
        '\\begin{center}',
        f'\\includegraphics[width=0.85\\textwidth]{{{feature_to_tex(fig_cv_name)}}}',
        '\\end{center}',
    )))
    fig_lines.append('\n'.join((
        '\\subsection{Predicted vs observed (EC11K, provided split 1)}',
        '\\begin{center}',
        f'\\includegraphics[width=0.85\\textwidth]{{{feature_to_tex(fig_scatter_name)}}}',
        '\\end{center}',
        )))
    fig_section = '\n\n'.join(fig_lines)
    tex = '\n'.join((
        '\\documentclass[10pt]{article}',
        '\\usepackage[a4paper,margin=25mm]{geometry}',
        '\\usepackage{booktabs}',
        '\\usepackage{amsmath}',
        '\\usepackage{hyperref}',
        '\\usepackage{graphicx}',
        f'\\title{{{feature_to_tex(report_title)}}}',
        '\\author{{}}',
        '\\date{{}}',
        '\\begin{document}',
        '\\maketitle',
        f'{ridge_section}',
        f'{fig_section}',
        f'{lime_section}',
        '\\end{document}',
    ))
    return tex

def build_pdf_from_tex(tex_path):
    """
    Compile a LaTeX file to PDF using pdflatex and return the resulting PDF path, raising an error if compilation fails.
    """
    outdir = tex_path.parent
    cmd = [
        'pdflatex', 
        '-interaction=nonstopmode', 
        '-halt-on-error', 
        tex_path.name,
    ]
    for _ in range(2):
        p = subprocess.run(
            cmd, 
            cwd=str(outdir), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                'pdflatex failed.\n\nSTDOUT:\n'
                + p.stdout 
                + '\n\nSTDERR:\n'
                + p.stderr
            )
    pdf_path = tex_path.with_suffix('.pdf')
    if not pdf_path.exists():
        raise RuntimeError('pdflatex reported success but PDF was not created.')
    return pdf_path

def write_figures(outdir, data, model, rows_cv, seed=0):
    """
    Generate and save cross-validation R² boxplots and a predicted-vs-observed scatter plot, returning their filenames.
    """
    if rows_cv is not None and len(rows_cv) > 0:
        r2_ec = [float(r['R2']) for r in rows_cv if r['dataset'] == 'EC11K']
        r2_mc = [float(r['R2']) for r in rows_cv if r['dataset'] == 'MC9K']
        fig1_path = outdir / 'fig_cv_r2_boxplot.pdf'
        plt.figure()
        plt.boxplot([r2_ec, r2_mc], labels=['EC11K', 'MC9K'])
        plt.ylabel(r'$R^2$')
        plt.title(r'Distribution of $R^2$ across repeated stratified CV folds')
        plt.tight_layout()
        plt.savefig(fig1_path)
        plt.close()
    else:
        fig1_path = None
    sub = data['EC11K']['splits'][1]
    X_train, y_train = sub['X_train'], sub['y_train']
    X_test, y_test = sub['X_test'], sub['y_test']
    m = clone(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    fig2_path = outdir / 'fig_ec11k_pred_vs_obs_split1.pdf'
    plt.figure()
    plt.scatter(y_test, y_pred, s=12)
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(r'Observed $\ln(\mathrm{IC50})$')
    plt.ylabel(r'Predicted $\ln(\mathrm{IC50})$')
    plt.title(r'EC11K ridge: predicted vs observed (provided split 1)')
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    return (
        fig1_path.name if fig1_path is not None else None,
        fig2_path.name,
    )

def build_report_title(cfg, *, prefix='Panobinostat: '):
    """
    Construct a report title from configuration flags indicating which analysis components were run.
    """
    parts = []
    if bool(cfg.get('run_compare_provided', False)) or bool(cfg.get('run_compare_cv', False)):
        parts.append('Ridge reproduction')
    if bool(cfg.get('run_lime', False)):
        parts.append('LIME')
    if not parts:
        return prefix.rstrip(': ')
    if len(parts) == 1:
        return prefix + parts[0]
    if len(parts) == 2:
        return prefix + parts[0] + ' and ' + parts[1]
    return prefix + ', '.join(parts[:-1]) + ', and ' + parts[-1]
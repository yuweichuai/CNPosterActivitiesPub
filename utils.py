"""Analysis-only helpers. Adapted for the pseudonymized review package."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
from scipy import stats
from itertools import combinations
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from review_setup import stata, Scalar, Matrix, ROOT


T_VALUE = 1.96

SEED = 710

FONT_SIZE = 22

main_blue = '#1F4E79'

blue = '#0072B2'

vermilion = '#D55E00'

bluish_green = '#009E73'

purple = '#766DF8'

golden_yellow = '#F0E442'

sky_blue = '#56B4E9'

soft_red = '#ef767a'

moderate_blue = '#456990'

moderate_cyan = '#49beaa'

single_color = '#007F66'

repeated_color = '#C7435B'

LW = 4

MZ = 12

fit_columns = ['user_follower_count', 'user_following_count', 'user_is_verified',
               'user_account_age_days', 'partisan_score', 'misinfo_exposure_score',
               'days_from_rollout', 'UserPostFrequency', 'UserPostPositive',
               'UserPostNegative', 'UserPostToxicity', 'UserPostMisleadingness',
               'UserPostConfidence', 'UserPostPolitics']

he_factors = [('user_follower_count', 'Account: followers', 'median'),
              ('user_following_count', 'Account: followees', 'median'),
              ('user_is_verified', 'Account: verified', 'binary'),
              ('user_account_age_days', 'Account: account age', 'median'),
              ('partisan_score', 'Account: political leaning', 'alignment'),
              ('misinfo_exposure_score', 'Account: misinfo exposure', 'median'),
              ('days_from_rollout', 'Correction: days from roll-out', 'median'),
              ('UserPostFrequency', 'Posting preference: frequency', 'median'),
              ('UserPostPositive', 'Posting preference: positive', 'median'),
              ('UserPostNegative', 'Posting preference: negative', 'median'),
              ('UserPostToxicity', 'Posting preference: toxicity', 'median'),
              ('UserPostMisleadingness', 'Posting preference: misleadingness', 'median'),
              ('UserPostConfidence', 'Posting preference: confidence', 'median'),
              ('UserPostPolitics', 'Posting preference: politics', 'median')]

he_content_factors = [('misleadingness', 'Content: misleadingness', 'UserPostMisleadingness', 'original_posts_narrow.parquet.gzip', 'days_from_display', 0.86),
                      ('positive', 'Content: positive', 'UserPostPositive', 'original_posts_wide.parquet.gzip', 'weeks_from_display', 0.5),
                      ('negative', 'Content: negative', 'UserPostNegative', 'original_posts_wide.parquet.gzip', 'weeks_from_display', 0.5),
                      ('toxicity', 'Content: toxicity', 'UserPostToxicity', 'original_posts_wide.parquet.gzip', 'weeks_from_display', 0.5),
                      ('confidence', 'Content: confidence', 'UserPostConfidence', 'original_posts_wide.parquet.gzip', 'weeks_from_display', 6.5),
                      ('politics_is', 'Content: politics', 'UserPostPolitics', 'original_posts_wide.parquet.gzip', 'weeks_from_display', 1)]

def get_source_corrections(window_days=28, post_correction_free=False, horizon_days=None):
    source_corrections = pd.read_parquet('data/source_post_features_processed.parquet')
    source_corrections = source_corrections.loc[~source_corrections[f'control_has_correction_{window_days}d']].reset_index(drop=True)
    if horizon_days:
        source_corrections = source_corrections.query(f'days_from_rollout>{horizon_days}').reset_index(drop=True)
    if post_correction_free:
        source_corrections = source_corrections.query(f'treated==0 or total_corrections==1 or '
                                                      f'(current_correction==1 & days2nextNote>{window_days}) '
                                                      f'or (total_corrections==current_correction & days2previousNote>{window_days}) '
                                                      f'or (days2previousNote>{window_days} & days2nextNote>{window_days})').reset_index(drop=True)
    else:
        source_corrections = source_corrections.query(f'treated==0 or current_correction==1 or days2previousNote>{window_days}').reset_index(drop=True)
    source_corrections['account_cluster'] = source_corrections['user_id'].astype('string')
    return source_corrections

def check_balance(data, covariates, treatment='treated', threshold=0.1, sort_smd=False, tab_export=False, tab_label='covariate_balance'):
    """Covariate balance in one unweighted dataset; treatment must be 0/1."""
    if not data[treatment].isin([0, 1]).all():
        raise ValueError(f'{treatment} must contain only 0 and 1, without missing values.')
    if data[treatment].nunique() != 2:
        raise ValueError('Both treatment and control groups are required.')

    rows = []

    for col in covariates:
        values = pd.to_numeric(data[col], errors='raise')
        values = values.replace([np.inf, -np.inf], np.nan)

        treated = values[data[treatment].eq(1)]
        control = values[data[treatment].eq(0)]

        n_t, n_c = treated.count(), control.count()
        mean_t, mean_c = treated.mean(), control.mean()
        sd_t, sd_c = treated.std(ddof=1), control.std(ddof=1)
        difference = mean_t - mean_c

        valid = values.dropna()
        is_binary = len(valid) > 0 and valid.isin([0, 1]).all()

        smd = np.nan
        if min(n_t, n_c) >= 2:
            if is_binary:
                pooled_sd = np.sqrt(
                    (mean_t * (1 - mean_t) + mean_c * (1 - mean_c)) / 2
                )
            else:
                pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2)

            if pooled_sd > 0:
                smd = difference / pooled_sd
            elif difference == 0:
                smd = 0.0
            else:
                smd = np.sign(difference) * np.inf

        rows.append({
            'variable': col,
            'type': 'binary' if is_binary else 'continuous',
            'n_treated': n_t,
            'n_control': n_c,
            'n_total': n_t + n_c,
            'missing_treated': treated.isna().sum(),
            'missing_control': control.isna().sum(),
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'sd_treated': sd_t,
            'sd_control': sd_c,
            'mean_difference': difference,
            'SMD': smd,
            'abs_SMD': abs(smd),
            'balanced': abs(smd) < threshold if pd.notna(smd) else pd.NA
        })

    if sort_smd:
        rows = pd.DataFrame(rows).sort_values('abs_SMD', ascending=False, na_position='last').reset_index(drop=True)
    else:
        rows = pd.DataFrame(rows)

    def export_balance_table(balance_table, tab_label):
        columns = [
            'variable', 'type', 'n_treated', 'n_control', 'SMD', 'abs_SMD', 'balanced'
        ]

        table = balance_table[columns].copy()
        table['balanced'] = table['balanced'].map({True: 'Yes', False: 'No'})

        table = table.rename(columns={
            'variable': 'Covariate',
            'type': 'Type',
            'n_treated': 'N treated',
            'n_control': 'N control',
            'n_total': 'N total',
            'SMD': 'SMD',
            'abs_SMD': 'Absolute SMD',
            'balanced': 'Balanced'
        })

        latex = table.to_latex(
            index=False,
            float_format='%.3f',
            formatters={
                col: lambda x: f'{x:,.0f}'
                for col in ['N treated', 'N control']
            },
            na_rep='--',
            escape=True,
            caption=(
                'Covariate balance between treatment and control groups. '
                'Counts refer to nonmissing observations for each covariate. '
                'Means of binary covariates represent proportions. '
                'Balance is defined as an absolute standardized mean difference '
                '(SMD) below 0.10.'
            ),
            label=f'tab:{tab_label}'
        )

        print(latex)
    if tab_export:
        export_balance_table(rows, tab_label)
    return rows

def _stata_condition(*conditions):
    """Combine bare expressions or 'if ...' qualifiers without changing OR precedence."""
    parts = []
    for condition in conditions:
        if condition:
            text = condition.strip()
            if text.startswith('if '):
                text = text[3:].strip()
            if text.startswith('&'):
                text = text[1:].strip()
            if text:
                parts.append(f'({text})')
    return 'if ' + ' & '.join(parts) if parts else ''

class StataDiD:
    def __init__(self, data, treat_col, unit_col, time_col, before_periods, after_periods):
        if not before_periods < 0 < after_periods:
            raise ValueError('Use a negative before_periods and positive after_periods.')
        data = data.copy()
        if data[[unit_col, time_col, treat_col]].isna().any().any():
            raise ValueError('Panel IDs, relative time, and treatment cannot be missing.')
        if data.duplicated([unit_col, time_col]).any():
            raise ValueError('Each event panel must have one row per relative period.')
        if not data[treat_col].isin([0, 1]).all():
            raise ValueError('Treatment must be coded 0/1.')
        if data.groupby(unit_col)[treat_col].nunique().gt(1).any():
            raise ValueError('Treatment must be constant within each event panel.')
        if not data[time_col].isin(list(range(before_periods, 0)) + list(range(1, after_periods + 1))).all():
            raise ValueError('Select the intended window and exclude period zero before initialization.')
        self.data = data
        self.treat_col, self.unit_col, self.time_col = treat_col, unit_col, time_col
        self.before_periods, self.after_periods = before_periods, after_periods
        for i in range(before_periods, 0):
            data[f'pre_{abs(i)}'] = data[time_col].eq(i).astype(int)
        for i in range(1, after_periods + 1):
            data[f'post_{i}'] = data[time_col].eq(i).astype(int)
        stata.pdataframe_to_data(data, force=True)
        # egen group works for numeric and string IDs; time remains numeric.
        stata.run(f'egen unitId = group({unit_col})')
        stata.run(f'xtset unitId {time_col}')
        for i in range(before_periods, -1):
            stata.run(f'gen treated_periods_pre{abs(i)} = {treat_col}*pre_{abs(i)}')
        for i in range(1, after_periods + 1):
            stata.run(f'gen treated_periods_post{i} = {treat_col}*post_{i}')
        stata.run(f'gen after = {time_col}>0')
        stata.run(f'gen treated_after = {treat_col}*after')
        stata.run(f'egen TFE = group({time_col})')

    def _run(self, reg, dev, rhs, random_unit=False, cvs=None, condition=None,
             cluster=False, appendix=None, pooled=False, weights=None):
        """cluster=True retains event clustering; cluster='user_id' clusters accounts."""
        # Stata weights belong before the comma and before any random-effects equation.
        weight_text = ''
        if weights is not None:
            if not isinstance(weights, str) or not re.fullmatch(
                r'\[(?:pw|iw|fw|aw|pweight|iweight|fweight|aweight)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\]',
                weights.strip()
            ):
                raise ValueError('Use weights="[pw=account_weight]" (or another supported Stata weight type).')
            weight_text = weights.strip()
        command = reg.split()[-1]
        if command == 'ppmlhdfe' and weight_text and not re.match(r'\[(?:pw|fw|pweight|fweight)\s*=', weight_text):
            raise ValueError('ppmlhdfe supports pweights or fweights; use [pw=account_weight].')
        options = (appendix or '').strip().lstrip(',').strip()
        if cluster and re.search(r'\bvce\s*\(', options):
            raise ValueError('Specify clustering through cluster OR appendix, not both.')
        cluster_col = 'unitId' if cluster is True else cluster
        vce = ''
        if cluster_col:
            if command == 'xtpoisson' and not random_unit:
                if cluster_col != 'unitId':
                    raise ValueError('Use ppmlhdfe for event fixed effects with account clustering.')
                vce = 'vce(robust)'
            else:
                vce = f'vce(cluster {cluster_col})'
        suffix = ''
        model_options = []
        if command == 'ppmlhdfe':
            if not re.search(r'\babsorb\s*\(', options):
                model_options.append('absorb(unitId TFE)')
        elif pooled:
            pass
        elif random_unit:
            suffix = ' || unitId:'
        else:
            model_options.append('fe')
        model_options += [value for value in (vce, options) if value]
        qualifier = _stata_condition(condition)
        text = f'{reg} {dev} {rhs} {cvs or ""} {qualifier} {weight_text}{suffix}'
        if model_options:
            text += ', ' + ' '.join(model_options)
        stata.run(text)

    def _event_rhs(self, aggregate=False):
        pre = range(-self.before_periods, 1, -1)
        post = range(1, self.after_periods + 1)
        terms = [self.treat_col]
        terms += [f'pre_{i}' for i in pre] + [f'post_{i}' for i in post]
        terms += [f'treated_periods_pre{i}' for i in pre]
        terms += ['treated_after'] if aggregate else [f'treated_periods_post{i}' for i in post]
        return ' '.join(terms)

    def multi_did(self, reg, dev, random_unit=False, cvs=None, condition=None, cluster=False, appendix=None, weights=None):
        self._run(reg, dev, self._event_rhs(), random_unit, cvs, condition, cluster, appendix, weights=weights)

    def multi_did_log(self, reg, dev, random_unit=False, cvs=None, condition=None, cluster=False, weights=None):
        self._run(reg, dev, self._event_rhs(), False, cvs, condition, cluster, pooled=random_unit, weights=weights)

    def multi_pre_did(self, reg, dev, random_unit=False, cvs=None, condition=None, cluster=False, weights=None):
        self._run(reg, dev, self._event_rhs(aggregate=True), random_unit, cvs, condition, cluster, weights=weights)

    def tw_did(self, reg, dev, random_unit=False, cvs=None, condition=None, cluster=False, appendix=None, weights=None):
        self._run(reg, dev, f'{self.treat_col} after treated_after i.TFE',
                  random_unit, cvs, condition, cluster, appendix, weights=weights)

    def tw_did_log(self, reg, dev, random_unit=False, cvs=None, condition=None, cluster=False, weights=None):
        self._run(reg, dev, f'{self.treat_col} after treated_after i.TFE',
                  False, cvs, condition, cluster, pooled=random_unit, weights=weights)

    def inter_did(self, reg, dev, interv, random_unit=False, cvs=None, condition=None, cluster=False, appendix=None, weights=None):
        """Fit separate treatment × after × moderator interactions jointly.

        interv accepts one numeric column name or a list/tuple of names.
        The first moderator retains inter_treated, inter_after, and
        inter_treated_after; subsequent moderators use suffixes _2, _3, etc.
        Includes each moderator's main effect and lower-order interactions,
        but no interactions between moderators.
        """
        if isinstance(interv, str):
            moderators = [interv]
        elif isinstance(interv, (list, tuple)):
            moderators = list(interv)
        else:
            raise ValueError('interv must be a column name or a list/tuple of column names.')
        if not moderators or any(
            not isinstance(var, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,31}', var)
            for var in moderators
        ):
            raise ValueError('Provide at least one plain numeric Stata column name in interv.')
        if len(set(moderators)) != len(moderators):
            raise ValueError('Interaction variables must be unique.')
        generated = {
            f'{base}{"" if i == 1 else "_" + str(i)}'
            for i in range(1, len(moderators) + 1)
            for base in ('inter_treated', 'inter_after', 'inter_treated_after')
        }
        if set(moderators) & (generated | {self.treat_col, 'after', 'treated_after', 'TFE', 'unitId'}):
            raise ValueError('Moderator names conflict with model or generated interaction variables.')
        # Validate all moderators before replacing any generated columns.
        for var in moderators:
            stata.run(f'confirm numeric variable {var}')
        extra_terms = []
        for i, var in enumerate(moderators, start=1):
            suffix = '' if i == 1 else f'_{i}'
            for base, factor in [('inter_treated', self.treat_col),
                                 ('inter_after', 'after'),
                                 ('inter_treated_after', 'treated_after')]:
                name = f'{base}{suffix}'
                stata.run(f'capture drop {name}')
                stata.run(f'gen double {name} = {var}*{factor}')
                if i > 1:
                    extra_terms.append(name)
        three_way = ['inter_treated_after'] + [
            name for name in extra_terms if name.startswith('inter_treated_after_')
        ]
        lower_order = [name for name in extra_terms if name not in three_way]
        rhs = (f'{self.treat_col} inter_treated after inter_after treated_after '
               + ' '.join(three_way + lower_order)
               + f' i.TFE {" ".join(moderators)}')
        self._run(reg, dev, rhs, random_unit, cvs, condition, cluster, appendix, weights=weights)

    #########################
    # Frequency Analysis
    #########################

    def freq_did(self, reg, dev, stat_type, freq_col, min_freq, max_freq, random_unit=False, condition=None, cip=95,
                 irr_or=0, trans=None, unit_group=True, graph=False, x_label='Frequency', y_label='ATT', ylims=None, output_name='frequency_did'):
        freq_means, freq_lls, freq_hls = [], [], []
        for fn in range(min_freq, max_freq+1):
            if fn==min_freq:
                print(fn, end=' ')
                self.tw_did(reg=f'qui {reg}', dev=dev, random_unit=random_unit,
                            condition=_stata_condition(f'{freq_col}=={fn}', condition))
            else:
                print(f'>={fn}', end=' ')
                self.tw_did(reg=f'qui {reg}', dev=dev, random_unit=random_unit,
                            condition=_stata_condition(f'{freq_col}>={fn}', condition))
            pos = Matrix.getColNames('e(b)').index('treated_after')
            freq_mean, freq_ll, freq_hl = get_coefs(reg=f'qui {reg}', start_line=pos, end_line=pos+1, stat_type=stat_type,
                                                cip=cip, irr_or=irr_or, trans=trans, unit_group=unit_group)
            freq_means.extend(freq_mean)
            freq_lls.extend(freq_ll)
            freq_hls.extend(freq_hl)
        if graph:
            plt.style.use('default')
            plt.rcParams['font.size'] = FONT_SIZE
            plt.figure(layout="constrained")
            x_ticks = np.arange(len(freq_means))
            plt.axhline(y=0, linestyle='--', color='lightgrey', linewidth=LW)
            plt.errorbar(x=x_ticks[0], y=freq_means[0],
                         yerr=[[freq_means[0] - freq_lls[0]], [freq_hls[0] - freq_means[0]]],
                         color=bluish_green, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.errorbar(x=x_ticks[1:len(x_ticks)-1], y=freq_means[1:len(x_ticks)-1],
                         yerr=[np.array(freq_means[1:len(x_ticks)-1]) - np.array(freq_lls[1:len(x_ticks)-1]),
                               np.array(freq_hls[1:len(x_ticks)-1]) - np.array(freq_means[1:len(x_ticks)-1])],
                         color=soft_red, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.errorbar(x=x_ticks[-1], y=freq_means[-1],
                         yerr=[[freq_means[-1] - freq_lls[-1]], [freq_hls[-1] - freq_means[-1]]],
                         color=soft_red, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.xticks(np.append(x_ticks[0], x_ticks[1:]),
                       [str(min_freq)] + [f'$\\geq${fn}' for fn in range(min_freq+1, max_freq+1)])
            plt.xlim(-0.5, len(freq_means)-0.5)
            if ylims is not None:
                plt.ylim(ylims[0], ylims[1])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(f'outputs/figures/{output_name}.pdf')
            plt.show()

    def cumc_did(self, reg, dev, stat_type, cum_col, min_cum, max_cum, random_unit=False, condition=None, cip=95,
                 irr_or=0, trans=None, unit_group=True, graph=False, x_label='Cumulative count', y_label='ATT', ylims=None, output_name='cumcount_did'):
        cumc_means, cumc_lls, cumc_hls = [], [], []
        for cn in range(min_cum, max_cum+1):
            if cn==min_cum:
                print(cn, end=' ')
                self.tw_did(reg=f'qui {reg}', dev=dev, random_unit=random_unit,
                            condition=_stata_condition(f'(({self.treat_col}==1 & {cum_col}=={cn}) | {self.treat_col}==0) & note_count>=2', condition))
            else:
                print(f'>={cn}', end=' ')
                self.tw_did(reg=f'qui {reg}', dev=dev, random_unit=random_unit,
                            condition=_stata_condition(f'(({self.treat_col}==1 & {cum_col}>={cn}) | {self.treat_col}==0) & note_count>={cn}', condition))
            pos = Matrix.getColNames('e(b)').index('treated_after')
            cumc_mean, cumc_ll, cumc_hl = get_coefs(reg=f'qui {reg}', start_line=pos, end_line=pos+1, stat_type=stat_type,
                                                cip=cip, irr_or=irr_or, trans=trans, unit_group=unit_group)
            cumc_means.extend(cumc_mean)
            cumc_lls.extend(cumc_ll)
            cumc_hls.extend(cumc_hl)
        if graph:
            plt.style.use('default')
            plt.rcParams['font.size'] = FONT_SIZE
            plt.figure(layout="constrained")
            x_ticks = np.arange(len(cumc_means))
            plt.axhline(y=0, linestyle='--', color='lightgrey', linewidth=LW)
            plt.errorbar(x=x_ticks[0], y=cumc_means[0],
                         yerr=[[cumc_means[0] - cumc_lls[0]], [cumc_hls[0] - cumc_means[0]]],
                         color=soft_red, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.errorbar(x=x_ticks[1:len(x_ticks)-1], y=cumc_means[1:len(x_ticks)-1],
                         yerr=[np.array(cumc_means[1:len(x_ticks)-1]) - np.array(cumc_lls[1:len(x_ticks)-1]),
                               np.array(cumc_hls[1:len(x_ticks)-1]) - np.array(cumc_means[1:len(x_ticks)-1])],
                         color=soft_red, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.errorbar(x=x_ticks[-1], y=cumc_means[-1],
                         yerr=[[cumc_means[-1] - cumc_lls[-1]], [cumc_hls[-1] - cumc_means[-1]]],
                         color=soft_red, marker='s', linestyle='', markersize=MZ, elinewidth=LW)
            plt.xticks(np.append(x_ticks[0], x_ticks[1:]),
                       [str(min_cum)] + [f'$\\geq${fn}' for fn in range(min_cum+1, max_cum+1)])
            plt.xlim(-0.5, len(cumc_means)-0.5)
            if ylims is not None:
                plt.ylim(ylims[0], ylims[1])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(f'outputs/figures/{output_name}.pdf')
            plt.show()

def get_coefs(reg, start_line, end_line, stat_type, cip=95, irr_or=0,
              trans=None, unit_group=True, full_return=False, unit_var=None,
              ppml_method='transform', account_var='account_cluster'):
    """Extract zero-based [start_line:end_line] coefficients from the fitted model.

    ppml_method='transform' exponentiates original confidence limits and retains
    original tests. 'nlcom' uses delta-method inference for exp(beta)-1.
    Existing menbreg/xt replay and return formats are preserved.
    By default, report distinct focal posts (unit_var or unitId) and
    accounts (account_var, default user_id) within e(sample).
    Set unit_group=False for regressions without these identifiers.
    """
    if not 0 < cip < 100:
        raise ValueError('cip must lie between 0 and 100.')
    if stat_type not in ('t', 'z') or irr_or not in (0, 1):
        raise ValueError('Use stat_type="t" or "z" and irr_or=0 or 1.')
    if ppml_method not in ('transform', 'nlcom'):
        raise ValueError('ppml_method must be transform or nlcom.')
    is_ppml = reg.split()[-1] == 'ppmlhdfe'
    replay_option = ''
    if irr_or and not is_ppml:
        if trans not in ('irr', 'or', 'eform'):
            raise ValueError('Specify trans="irr", "or", or "eform".')
        replay_option = trans
    stata.run(f'{reg}, {replay_option} level({cip})')
    res = np.asarray(stata.get_return()['r(table)']).T.copy()
    names = Matrix.getColNames('e(b)')
    if not 0 <= start_line < end_line <= len(res):
        raise ValueError(f'Coefficient range must satisfy 0 <= start < end <= {len(res)}.')
    n_obs = int(Scalar.getValue('e(N)'))
    unit_text = ''
    if unit_group:
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO

        def count_ids(column):
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', column):
                raise ValueError('Supply a single Stata variable name for each ID.')
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                stata.run(f"""
                    quietly assert !missing({column}) if e(sample)
                    tempvar id_tag
                    quietly egen `id_tag' = tag({column}) if e(sample)
                    quietly count if `id_tag' == 1
                """, echo=False)
            return int(stata.get_return()['r(N)'])

        # Count distinct IDs in the actual estimation sample, not absorbed DoF.
        n_posts = count_ids(unit_var or 'unitId')
        n_accounts = count_ids(account_var)
        unit_text = f', #Posts: {n_posts:,}, #Accounts: {n_accounts:,}'
    print(f'#Obs: {n_obs:,}{unit_text}')
    means, std_errs, statistics, pvals, clls, chls = [], [], [], [], [], []
    for row in range(start_line, end_line):
        line = res[row].copy()
        name = names[row]
        if not np.isfinite(line[1]) or line[1] <= 0:
            raise ValueError(f'{name} has no positive finite SE; check omission or identification.')
        current_stat = stat_type
        if is_ppml and irr_or:
            if ppml_method == 'nlcom':
                stata.run(f'quietly nlcom (effect: exp(_b[{name}])-1), level({cip})')
                line = np.asarray(stata.get_return()['r(table)'])[:, 0].copy()
                mean, se, lower, upper = line[0], line[1], line[4], line[5]
                current_stat = 'z'
            else:
                mean, se = np.expm1(line[0]), np.exp(line[0]) * line[1]
                lower, upper = np.expm1(line[4]), np.expm1(line[5])
        else:
            shift = 1 if irr_or else 0
            mean, se = line[0] - shift, line[1]
            lower, upper = line[4] - shift, line[5] - shift
        stat = (f'$t({line[6]:g})={line[2]:.3f}$' if current_stat == 't'
                else f'$z={line[2]:.3f}$')
        pval = f'$p={line[3]:.3f}$' if line[3] >= .001 else '$p<0.001$'
        print(f'{name}: Estimate $= {mean:.3f}$, {stat}, {pval}; '
              f'{cip}\%~CI: $[{lower:.3f}, {upper:.3f}]$')
        means.append(mean)
        std_errs.append(se)
        statistics.append(stat)
        pvals.append(pval)
        clls.append(lower)
        chls.append(upper)
    if full_return:
        return means, std_errs, statistics, pvals, clls, chls
    return means, clls, chls

def _mean_ci_single(data, score='misleadingness', cluster=None,
            method='t', confidence=0.95, n_boot=10_000, seed=SEED):
    """
    CI for an observation-weighted mean.

    data: Series or DataFrame
    cluster: account/cluster column, or None for independent observations
    method: 't' or 'bootstrap'
    """
    if method not in {'t', 'bootstrap'}:
        raise ValueError("method must be 't' or 'bootstrap'")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if method == 'bootstrap' and (
        not isinstance(n_boot, (int, np.integer)) or n_boot < 2
    ):
        raise ValueError("n_boot must be an integer >= 2")

    if isinstance(data, pd.Series):
        if cluster is not None:
            raise ValueError("Pass a DataFrame when specifying cluster.")
        frame = data.to_frame(name=score)
    else:
        columns = [score] if cluster is None else [score, cluster]
        frame = data[columns].copy()

    frame[score] = pd.to_numeric(frame[score], errors='raise')
    frame = frame.replace({score: {np.inf: np.nan, -np.inf: np.nan}})
    frame = frame.dropna()

    x = frame[score].to_numpy(dtype=float)
    n = len(x)
    mean = x.mean() if n else np.nan
    lower = upper = se = np.nan

    if cluster is None:
        sums = x
        counts = np.ones(n, dtype=int)
    else:
        grouped = frame.groupby(cluster, observed=True)[score].agg(['sum', 'count'])
        sums = grouped['sum'].to_numpy(dtype=float)
        counts = grouped['count'].to_numpy()

    # Independent sampling units: observations or clusters.
    g = len(sums)

    if g >= 2:
        if method == 't':
            # For clusters, this is the intercept-only cluster-robust SE
            # with finite-sample correction. Without clusters, it is s/sqrt(n).
            residual_sums = sums - counts * mean
            se = np.sqrt(g / (g - 1) * np.sum(residual_sums**2)) / n
            margin = stats.t.ppf((1 + confidence) / 2, df=g - 1) * se
            lower, upper = mean - margin, mean + margin

        else:
            rng = np.random.default_rng(seed)
            boot_means = np.empty(n_boot)

            for b in range(n_boot):
                idx = rng.integers(0, g, size=g)
                boot_means[b] = sums[idx].sum() / counts[idx].sum()

            se = boot_means.std(ddof=1)
            alpha = (1 - confidence) / 2
            lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])

    return pd.Series({
        'n': n,
        'n_units': g,
        'mean': mean,
        'se': se,
        'ci_lower': lower,
        'ci_upper': upper
    })

def mean_ci(data, score='misleadingness', cluster=None, group=None,
            method='t', confidence=0.95, n_boot=10_000, seed=SEED):

    if group is None:
        return _mean_ci_single(
            data, score=score, cluster=cluster, method=method,
            confidence=confidence, n_boot=n_boot, seed=seed
        )

    # Group comparisons below use regression-based t tests.
    if method != 't':
        raise ValueError("Grouped comparisons currently support method='t' only.")

    columns = list(dict.fromkeys(
        [score, group] + ([cluster] if cluster else [])
    ))
    df = data[columns].copy()
    df[score] = pd.to_numeric(df[score], errors='raise')
    df = df.replace({score: {np.inf: np.nan, -np.inf: np.nan}}).dropna()

    summary = pd.DataFrame({
        name: _mean_ci_single(
            subset, score=score, cluster=cluster,
            method='t', confidence=confidence
        )
        for name, subset in df.groupby(group, observed=True, sort=False)
    }).T

    # Numeric group codes avoid problems with spaces or punctuation in labels.
    codes, labels = pd.factorize(df[group], sort=False)
    k = len(labels)
    if k < 2:
        raise ValueError("At least two observed groups are required.")

    regression_data = pd.DataFrame({
        'y': df[score].to_numpy(dtype=float),
        'g': codes
    })

    regression = smf.ols('y ~ 0 + C(g)', data=regression_data)

    if cluster is not None:
        if df[cluster].nunique() < 2:
            raise ValueError("At least two clusters are required.")
        model = regression.fit(
            cov_type='cluster',
            cov_kwds={
                'groups': df[cluster].to_numpy(),
                'use_correction': True
            },
            use_t=True
        )
    else:
        model = regression.fit(cov_type='HC3', use_t=True)

    rows = []
    for i, j in combinations(range(k), 2):
        contrast = np.zeros(k)
        contrast[i], contrast[j] = 1, -1

        test = model.t_test(contrast)
        lower, upper = test.conf_int(alpha=1 - confidence)[0]

        rows.append({
            'group_a': labels[i],
            'group_b': labels[j],
            'difference': np.asarray(test.effect).item(),  # A minus B
            'ci_lower': lower,
            'ci_upper': upper,
            'p_value': np.asarray(test.pvalue).item()
        })

    differences = pd.DataFrame(rows)
    differences['p_holm'] = multipletests(
        differences['p_value'], method='holm'
    )[1]
    differences['significant'] = (
        differences['p_holm'] < 1 - confidence
    )

    # Joint null: every group mean equals the first group mean.
    contrasts = np.zeros((k - 1, k))
    contrasts[:, 0] = -1
    contrasts[np.arange(k - 1), np.arange(1, k)] = 1
    overall_test = model.f_test(contrasts)

    overall = pd.Series({
        'F': np.asarray(overall_test.fvalue).item(),
        'df_num': overall_test.df_num,
        'df_denom': overall_test.df_denom,
        'p_value': np.asarray(overall_test.pvalue).item()
    })

    return {
        'summary': summary,
        'differences': differences,
        'overall': overall
    }

def att_plot(ts_start, ts_end, before_period, mp_means_att, mp_lls_att, mp_hls_att, agg_mean_att, agg_ll_att, att_hl_att,
             legend_label, x_label, y_label, output_name, ylims=None, fig_size=None, legend_ncol=2, legend_title=''):
    plt.style.use('default')
    plt.rcParams['font.size'] = FONT_SIZE
    if fig_size is not None:
        fig_plot, ax_plot = plt.subplots(figsize=fig_size, layout='constrained')
    else:
        fig_plot, ax_plot = plt.subplots(layout='constrained')
    ax_plot.axvspan(-before_period-1, 0, color='lightgrey', alpha=0.5)
    plt.axhline(y=0, linestyle='--', color='lightgrey', linewidth=LW)
    plt.errorbar(np.arange(ts_start+0.5, -0.5, 1), np.array(mp_means_att[:-ts_start-1]),
                 yerr=[np.array(mp_means_att[:-ts_start-1]) - np.array(mp_lls_att[:-ts_start-1]),
                       np.array(mp_hls_att[:-ts_start-1]) - np.array(mp_means_att[:-ts_start-1])],
                 marker='_', linestyle='', color=purple, markersize=MZ, mew=LW, linewidth=LW, label=legend_label)
    plt.errorbar(-0.5, 0, yerr=0, marker='_', linestyle='', color=purple, markersize=MZ, mew=LW, linewidth=LW)
    plt.errorbar(np.arange(0.5, ts_end+0.5, 1), np.array(mp_means_att[-ts_start-1:]),
                 yerr=[np.array(mp_means_att[-ts_start-1:]) - np.array(mp_lls_att[-ts_start-1:]),
                       np.array(mp_hls_att[-ts_start-1:]) - np.array(mp_means_att[-ts_start-1:])],
                 marker='_', linestyle='', color=purple, markersize=MZ, mew=LW, linewidth=LW)
    plt.errorbar(ts_end+1, agg_mean_att[0], yerr=[[agg_mean_att[0]-agg_ll_att[0]], [att_hl_att[0]-agg_mean_att[0]]],
                 marker='_', linestyle='', color=vermilion, markersize=MZ, mew=LW, linewidth=LW, label='Aggregated')
    plt.xticks(np.arange(ts_start, ts_end+2),
               [f'$-${abs(i)}' for i in np.arange(ts_start, 0)]+[i for i in np.arange(0, ts_end+1)]+['Agg'])
    plt.xlim(ts_start-0.2, ts_end+1.8)
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right', ncol=legend_ncol, fontsize=FONT_SIZE-3, title=legend_title)
    plt.savefig(f'outputs/figures/{output_name}.pdf')
    plt.show()

def prepare_content_count(filename, var, tcol, threshold_external):
    """Load the processed counts used by the original analysis.

    filename is a source-dataset alias, retained for traceability. The manifest
    maps it to an event-period count table; original post records are not needed.
    Changing a classifier threshold requires new processed input, not relabeling
    the supplied counts. Missing count cells remain missing until the original
    notebook's explicit fillna(0) step.
    """
    manifest = json.loads((ROOT / 'data/content_counts/manifest.json').read_text())
    matches = [m for m in manifest if m['source_alias'] == filename and m['variable'] == var and m['time_column'] == tcol]
    if len(matches) != 1 or float(matches[0]['threshold']) != float(threshold_external):
        raise ValueError('No processed count table for the requested dataset, variable, horizon and threshold.')
    return pd.read_parquet(ROOT / matches[0]['path'])


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import pingouin as pg
import scipy.stats as scs
from scipy.stats import shapiro
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import seaborn as sns


# def get_sigma(metric: str, p:float, std:pd.DataFrame) -> float:
#     """
#     :param metric: name metric
#     :return: standard deviation
#     """
#     if metric == 'CR' or metric == 'Discrete':
#         # print("Введите базовое значение (конверсии)")
#         p = float(input())
#         return (p*(1-p))**0.5
#     else:
#         # print("Введите стандартное отклонение(std)")
#         std = float(input())
#         return std


# def effect_size_func(mu_control: float, mu_experiment: float, sigma: float) -> float:
#     """
#     :param mu_control: mean metric in control group
#     :param mu_experiment: mean metric in experiment group
#     :param sigma: Standard deviation
#     :return: effect_size
#     """
#     return np.abs(mu_experiment - mu_control) / sigma


# def min_sample_size(effect_size: float, power=0.8, alpha=0.05) -> int:
#     """
#     :param effect_size: from effect_size_func
#     :param power: power test
#     :param alpha: significant level
#     :return: minimal sample size for test
#     """
#     # standard normal distribution to determine z-values
#     standard_norm = scs.norm(0, 1)

#     # find Z_beta from desired power
#     z_beta = standard_norm.ppf(power)

#     # find Z_alpha
#     z_alpha = standard_norm.ppf(1-alpha/2)

#     # average of probabilities from both groups

#     sample_size = (2 * (z_beta + z_alpha)**2 / effect_size**2)

#     return round(sample_size)

def calculate_sample_size(delta, sigma, alpha=0.05, beta=0.8,  ratio=1):
    # alpha - уровень значимости
    # beta - мощность теста
    # delta - ожидаемый эффект
    # sigma - стандартное отклонение
    # ratio - отношение размеров выборок
    if ratio == 0:
        raise ValueError("Ratio should not be zero")
    # Значение Z для уровня значимости и мощности теста
    Z_alpha = abs(norm.ppf(1-alpha/2))
    Z_beta = abs(norm.ppf(beta))

    # Расчет размера выборки для большей группы
    n1 = ((Z_alpha + Z_beta)**2 * (sigma**2) * (1 + 1/ratio)) / delta**2

    # Расчет размера выборки для меньшей группы
    n2 = n1 / ratio

    # Округление до ближайшего целого числа и возвращение результата
    return math.ceil(n1), math.ceil(n2)

def clear_data_group_columns(df: pd.DataFrame, column_group: str):
    message = ''
    if df[column_group].nunique() > 2:
        message = f'Ошибка в записи данных: значений в столбце {column_group} больше 2-х.'
        return pd.DataFrame(), message
    elif df[column_group].nunique() < 2:
        message = f'Ошибка в записи данных: значений в столбце {column_group} меньше 2-х.'
        return pd.DataFrame(), message
    if df[column_group].isna().sum():
        message += f"Процент пропущенных значений в столбце {column_group} " + \
                  str((df[column_group].isna().sum())/(df.shape[0]) * 100) + "%."
        df.dropna(subset=[column_group], inplace=True)
        message += f"Пропущенные значения были удалены из столбца {column_group}."
    return df, message

def clear_data(df: pd.DataFrame, column_group: str, column_value: str, type_data: str):
    """
    :param df: DataFrame
    :param column_group: name column with group
    :param column_value: name column with value
    :return: clear DataFrame
    """
    message = ''
#   fix types
    if type_data == 'Continuous' or type_data == 'Ranking':
        if df[column_value].dtype == 'O':
            try:
                df[column_value] = df[column_value].dropna().apply(lambda x: x.replace(',', '.'))
                df[column_value] = df[column_value].astype('float')
            except:
                message = f'Ошибка в записи данных: значения столбца {column_value} имеют неверный формат.'
                return pd.DataFrame(), message, False

        elif df[column_value].dtype == 'int64' or df[column_value].dtype == 'int32':
            df[column_value] = df[column_value].astype('float')
        elif df[column_value].dtype != 'float':
            message = f'Ошибка в записи данных: значения столбца {column_value} имеют неверный формат.'
            return pd.DataFrame(), message, False

        
        
    if df[column_value].isna().sum():
        message += f"Процент пропущенных значений в столбце {column_value} " + \
                   str(round((df[column_value].isna().sum())/(df.shape[0]) * 100, 2)) + "%."
#         Введите (без кавычек):\n
#         'del', если удалить\n
#         'min', если приравнять к минимальному ({df[column_value].min()})\n
#         'max', если приравнять к максимальному ({df[column_value].max()})\n
#         'avg', если приравнять к среднему ({df[column_value].mean()})\n
#         'median', если приравнять к медиане ({df[column_value].median()})""")
#         <digit>, если заменить на <digit>
    have_missing = df[column_value].isna().sum() > 0

        
    # работа с пропусками
    return df, message, have_missing


def missing_values(df: pd.DataFrame, column_group: str, column_value: str, str_val='del'):
    message = ''
    if str_val == 'Удалить':
        df[column_value] = df[column_value].dropna()
        message += "Пропущенные значения были удалены."
    elif str_val == 'Минимальное':
        df[column_value] = df[column_value].fillna(df[column_value].min())
        message += "Пропущенные значения были заменены на минимальное."
    elif str_val == 'Максимальное':
        df[column_value] = df[column_value].fillna(df[column_value].max())
        message += "Пропущенные значения были заменены на максимальное."
    elif str_val == 'Среднее':
        df[column_value] = df[column_value].fillna(df[column_value].mean())
        message += "Пропущенные значения были заменены на среднее."
    elif str_val == 'Медиану':
        df[column_value] = df[column_value].fillna(df[column_value].median())
        message += "Пропущенные значения были заменены на медиану."
    elif str_val.isdigit():
        df[column_value] = df[column_value].fillna(float(str_val))
        message += f"Пропущенные значения были заменены на {str_val}."
    return df, message


def f_test(group_1: pd.DataFrame, group_2: pd.DataFrame) -> float:
    """
    :param group_1: DataFrame with group 1
    :param group_2: DataFrame with group 2
    :return: p_value f_test
    """
    x = np.array(group_1)
    y = np.array(group_2)
    f = np.var(x, ddof=1)/np.var(y, ddof=1)  #calculate F test statistic
    dfn = x.size - 1  #define degrees of freedom numerator
    dfd = y.size - 1  #define degrees of freedom denominator
    p = 1 - scs.f.cdf(f, dfn, dfd)  #find p-value of F test statistic
    return p


def get_outliers(data: pd.DataFrame, name_column_metric: str):
    """
    :param data: DataFrame
    :param name_column_metric: name column metric
    :param n: choice for outliers
    :return: DataFrame with fix outliers
    """
    x_1 = data[name_column_metric].quantile(0.25) - 1.5 * (
            data[name_column_metric].quantile(0.75) - data[name_column_metric].quantile(0.25))
    x_2 = data[name_column_metric].quantile(0.75) + 1.5 * (
                data[name_column_metric].quantile(0.75) - data[name_column_metric].quantile(0.25))

    outliers = list(data[(data[name_column_metric] < x_1) | (data[name_column_metric] > x_2)][name_column_metric])
    data_outliers = data[data[name_column_metric].isin(outliers)]
    outliers_count = round(data_outliers.shape[0] / (data.shape[0]) * 100, 2)
    message = 'процент выбросов ' + str(outliers_count) + "%."
    is_outliers = False
    if outliers_count > 0:
        is_outliers = True
    return message, is_outliers

def change_outliers(df: pd.DataFrame, name_column_metric: str, name_column_group: str, n: str):
    # Введите номер варианта:
    # 1) Удалить выбросы;
    # 2) Заменить на максимальные и минимальные значение;
    # 3) Заменить на среднее;
    # 4) Заменить на медиану;
    # 5) Оставить выбросы
    # """)
    df_result = pd.DataFrame()
    for group in df[name_column_group].unique():
        data = df[df[name_column_group] == group]
        x_1 = data[name_column_metric].quantile(0.25) - 1.5 * (
            data[name_column_metric].quantile(0.75) - data[name_column_metric].quantile(0.25))
        x_2 = data[name_column_metric].quantile(0.75) + 1.5 * (
                data[name_column_metric].quantile(0.75) - data[name_column_metric].quantile(0.25))
        outliers = list(data[(data[name_column_metric] < x_1) | (data[name_column_metric] > x_2)][name_column_metric])
        max_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].max()
        min_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].min()
        flag_outliers = False
        if n != 'Удалить' and n != 'Максимальное и минимальное' and n != 'Среднее' and n != 'Медиану' and n != 'Оставить':
            raise ValueError("Нет предложенных вариантов")
        elif n == 'Удалить':
            message = "\nВыбросы были удалены."
            data =  data[~data[name_column_metric].isin(outliers)]
        elif n == 'Максимальное и минимальное':
            data[name_column_metric] = data[name_column_metric].\
                apply(lambda x: min_value if x < min_value else max_value if x > max_value else x)
            message = "\nВыбросы были заменены на максимальное и минимальное значение."
        elif n == 'Среднее':
            mean_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].mean()
            data[name_column_metric] = data[name_column_metric].\
                apply(lambda x: mean_value if x < min_value else mean_value if x > max_value else x)
            message = "\nВыбросы были заменены на среднее."
        elif n == 'Медиану':
            median_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].median()
            data[name_column_metric] = data[name_column_metric].\
                apply(lambda x: median_value if x < min_value else median_value if x > max_value else x)
            message = "\nВыбросы были заменены на медиану."
        else:
            flag_outliers = True
            message = "\nВыбросы остались."
        df_result = pd.concat([df_result, data])
    return df_result, message, flag_outliers


def get_bootstrap(
    data_column_1,  # числовые значения первой выборки
    data_column_2,  # числовые значения второй выборки
    boot_it=3000,  # количество бутстрап-подвыборок
    statistic=np.mean,  # интересующая нас статистика
    bootstrap_conf_level=0.95  # уровень значимости
):
    boot_data = []
    for i in range(boot_it):  # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1),
            replace=True  # параметр возвращения
        ).values

        samples_2 = data_column_2.sample(
            len(data_column_1),
            replace=True
        ).values

        boot_data.append(statistic(samples_1)-statistic(samples_2))  # mean() - применяем статистику

    pd_boot_data = pd.DataFrame(boot_data)

    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])

    p_1 = norm.cdf(
        x=0,
        loc=np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_2 = norm.cdf(
        x=0,
        loc=-np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2

    # Визуализация
    # _, _, bars = plt.hist(pd_boot_data[0], bins=50)
    # for bar in bars:
    #     if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
    #         bar.set_facecolor('red')
    #     else:
    #         bar.set_facecolor('grey')
    #         bar.set_edgecolor('black')
    #
    # plt.style.use('ggplot')
    # plt.vlines(quants, ymin=0, ymax=50, linestyle='--')
    # plt.xlabel('boot_data')
    # plt.ylabel('frequency')
    # plt.title("Histogram of boot_data")
    # plt.show()

    return {"boot_data": boot_data,
            "quants": quants,
            "p_value": p_value}


def get_bootstrap_for_aa(
    data_column_1,  # числовые значения первой выборки
    data_column_2,  # числовые значения второй выборки,
    type_data,
    boot_it=2000,  # количество бутстрап-подвыборок
    statistic=np.mean,  # интересующая нас статистика
    bootstrap_conf_level=0.95  # уровень значимости
):
    p_val_data = []
    for i in range(boot_it):  # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1),
            replace=True  # параметр возвращения
        ).values

        samples_2 = data_column_2.sample(
            len(data_column_1),
            replace=True
        ).values
        if type_data == 'Continuous':
            t_test_result = pg.ttest(samples_1, samples_2)
            p_val = round(t_test_result['p-val'][0], 4)
            p_val_data.append(p_val) 
        elif type_data == 'Discrete':
            df = pd.DataFrame({'name_column_group':['a1' for i in range(len(data_column_1))] + ['a2' for i in range(len(data_column_2))], 'name_column_metric':list(samples_1) + list(samples_2)})
            _, _, stats = pg.chi2_independence(df, x='name_column_group', y='name_column_metric')
            p_val = stats.round(3).query('test == "pearson"')['pval'][0]
            p_val_data.append(p_val)
        else:
            p_val = pg.mwu(samples_1, samples_2,
                           alternative='two-sided')['p-val'][0]
            p_val_data.append(p_val)

    return p_val_data


def get_p_value(metric: str, df: pd.DataFrame, name_column_group: str, name_column_metric: str, have_outliers: bool):
    """
    :param metric: name metric
    :param df: DataFrame
    :param name_column_group: name column group
    :param name_column_metric: name column metric
    :return: p_value, power
    """
    group_names = list(df[name_column_group].unique())
    name_1 = group_names[0]
    name_2 = group_names[1]
    df_group_1 = df[df[name_column_group] == name_1]
    df_group_2 = df[df[name_column_group] == name_2]
    message = ""
    if metric == 'Discrete':
        _, _, stats = pg.chi2_independence(df, x=name_column_group, y=name_column_metric)
        p_value = stats.round(3).query('test == "pearson"')['pval'][0]
        power = stats.round(3).query('test == "pearson"')['power'][0]
        return p_value, power, message

    if metric == 'Continuous' or metric == 'Ranking':
        distribution = 'normal'
        for gr in group_names:
            df_group = df[df[name_column_group] == gr]
            p_val_shapiro = shapiro(df_group[name_column_metric]).pvalue
            if p_val_shapiro < 0.05:
                distribution = 'abnormal'
                break
        if distribution == 'normal':
            message += '\nДанные в группах из нормального распределения'
        else:
            message += '\nДанные в группах не из нормального распределения'
        dispersion = 'equal'
        if distribution == 'normal':
            if f_test(df_group_1[name_column_metric], df_group_2[name_column_metric]) < 0.05:
                dispersion = 'unequal'
        if dispersion == 'equal':
            message += '\nДисперсии в группах равны'
        else:
            message += '\nДисперсии в группах различны'
        if metric != 'Ranking' and distribution == 'normal' and dispersion == 'equal' and not have_outliers:
            t_test_result = pg.ttest(df[df[name_column_group] == name_1][name_column_metric],
                                     df[df[name_column_group] == name_2][name_column_metric])
            return round(t_test_result['p-val'][0], 4), round(t_test_result['power'][0], 4), message
        if metric != 'Ranking':
            return round(get_bootstrap(df_group_1[name_column_metric],
                                       df_group_2[name_column_metric])['p_value'], 4), round(np.nan, 4), message
        else:
            p_val = pg.mwu(df_group_1[name_column_metric], df_group_2[name_column_metric],
                           alternative='two-sided')['p-val'][0]
            return round(p_val, 4), round(np.nan, 4), message

def get_hist(df: pd.DataFrame, name_column_group: str, name_column_metric: str, type_metric: str):
    binwidth = 0
    if type_metric == 'Continuous':
        for group in df[name_column_group].unique():
            binwidth = np.max([binwidth, 2*(df[df[name_column_group]==group][name_column_metric].quantile(0.75) - df[df[name_column_group]==group][name_column_metric].quantile(0.25))/df[df[name_column_group]==group].shape[0]**(1/3)])
        
        fig, ax = plt.subplots(figsize=(5, 4))
        image = sns.histplot(data = df, x=name_column_metric, hue=name_column_group, binwidth=binwidth, stat='density', common_norm=False, ax=ax).get_figure()
        plt.title(f"Гистограмма плотности для {name_column_metric}")
        plt.close(image)
        
    if type_metric == 'Discrete':
        pct2 = (df.groupby([name_column_group, name_column_metric]).size() / df.groupby([name_column_group]).size()).reset_index().rename({0:'Процент'}, axis=1)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = sns.barplot(x=name_column_metric, hue=name_column_group, y='Процент', data=pct2, ax=ax).get_figure()
        plt.title(f"Barplot для {name_column_metric}")
        plt.figure(figsize=(5,4))
        plt.close(image)
        
    if type_metric == 'Ranking':
        pass
    return image

def get_kde(df: pd.DataFrame, name_column_group: str, name_column_metric: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = sns.kdeplot(x=name_column_metric, data=df, hue=name_column_group, common_norm=False).get_figure()
    plt.title(f"Ядерная оценка плотности для {name_column_metric}")
    plt.figure(figsize=(5,4))
    plt.close(image)
    return image

def get_qq(df: pd.DataFrame, name_column_group: str, name_column_metric: str):
    values = df[name_column_metric].values
    df_pct = pd.DataFrame()
    for group in df[name_column_group].unique():
        df_pct[f'q_{group}'] = np.percentile(df.loc[df[name_column_group]==group, name_column_metric].values, range(100))
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.scatter(x=df_pct.columns[0], y=df_pct.columns[1], data=df_pct, label='Actual fit')
    image = sns.lineplot(x=df_pct.columns[0], y=df_pct.columns[0], data=df_pct, color='r', label='Line of perfect fit', ax=ax).get_figure()
    plt.xlabel(f'Quantile {df_pct.columns[0]} group')
    plt.ylabel(f'Quantile {df_pct.columns[1]} group')
    plt.legend()
    plt.title(f"QQ plot для {name_column_metric}")
    plt.figure(figsize=(5,4))
    plt.close(image)
    return image
    
def get_boxplot(df: pd.DataFrame, name_column_group: str, name_column_metric: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = sns.boxplot(data=df, x=name_column_group, y=name_column_metric, ax=ax).get_figure()
    plt.title(f"Ящик с усами для {name_column_metric}")
    plt.figure(figsize=(5,4))
    plt.close(image)
    return image

def get_image(df: pd.DataFrame, name_column_group: str, cont_name_column_metric_list: list, desc_name_column_metric_list: list):
    list_image_cont_hist = []
    list_image_cont_kde = []
    list_image_cont_qq = []
    list_image_cont_boxplot = []
    list_image_desc = []
    for name_column_cont in cont_name_column_metric_list:
        list_image_cont_hist.append(get_hist(df, name_column_group, name_column_cont, 'Continuous'))
        list_image_cont_kde.append(get_kde(df, name_column_group, name_column_cont))
        list_image_cont_qq.append(get_qq(df, name_column_group, name_column_cont))
        list_image_cont_boxplot.append(get_boxplot(df, name_column_group, name_column_cont))
    for name_column_desc in desc_name_column_metric_list:
        list_image_desc.append(get_hist(df, name_column_group, name_column_desc, 'Discrete'))
    return list_image_cont_hist, list_image_cont_kde, list_image_cont_qq, list_image_cont_boxplot, list_image_desc

def aa_p_val(df: pd.DataFrame, name_column_group: str, name_column_value: str, type_value:str):
    result = []
    result = get_bootstrap_for_aa(df[df[name_column_group] == df[name_column_group].unique()[0]][name_column_value], df[df[name_column_group] == df[name_column_group].unique()[1]][name_column_value],type_value)
    df_p_val = pd.DataFrame({name_column_value:result})
    fig, ax = plt.subplots(figsize=(5, 4))
    image = sns.histplot(data = df_p_val, x=name_column_value, ax=ax, bins=100).get_figure()
    plt.title(f"Гистограмма распределения p_value для {name_column_value}")
    plt.close(image)
    return image
    
def get_image_aa(df: pd.DataFrame, name_column_group: str, cont_name_column_metric_list: list, desc_name_column_metric_list: list, rank_name_column_metric_list: list):
    list_image_cont = []
    list_image_desc = []
    list_image_rank = []
    for name_column_cont in cont_name_column_metric_list:
        list_image_cont.append(aa_p_val(df, name_column_group, name_column_cont, "Continuous"))
    for name_column_desc in desc_name_column_metric_list:
        list_image_desc.append(aa_p_val(df, name_column_group, name_column_desc, "Discrete"))
    for name_column_rank in list_image_rank:
        list_image_rank.append(aa_p_val(df, name_column_group, name_column_rank, "Ranking"))
    return list_image_cont, list_image_desc, list_image_rank
    
# def get_conclusion(df: pd.DataFrame, name_column_group: str, name_column_metric: str, p_val: float):
#     """
#     :param df: DataFrame
#     :param name_column_group: name column group
#     :param name_column_metric: name column metric
#     :param p_val: p_value
#     """
#     group_names = list(df[name_column_group].unique())
#     name_1 = group_names[0]
#     name_2 = group_names[1]
#     df_group_1 = df[df[name_column_group] == name_1]
#     df_group_2 = df[df[name_column_group] == name_2]
#     mean_group_1 = df_group_1[name_column_metric].mean()
#     mean_group_2 = df_group_2[name_column_metric].mean()
#     if p_val < 0.05:
#         return f'''Среднее в группе {name_1}: {round(mean_group_1, 4)}.\nСреднее в группе {name_2}: {round(mean_group_2, 4)}.\nРазличия в средних статистически значимы.'''
#     else:
#         return f'''Среднее в группе {name_1}: {round(mean_group_1, 4)}.\nСреднее в группе {name_2}: {round(mean_group_2, 4)}.\nРазличия в средних статистически незначимы.'''

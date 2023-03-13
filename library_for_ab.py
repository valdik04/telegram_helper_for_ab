import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import pingouin as pg
import scipy.stats as scs
from scipy.stats import shapiro
from scipy.stats import norm
import matplotlib as plt


def what_is_metric() -> str:
    """
    :return: name metric
    """
    while True:
        # print("""Подскажи, какую метрику мы будем отслеживать. Выбери вариант ответа из предложенных.
        # Введи число или слово STOP, если хочешь начать сначала.
        # 1) CR(CTR и т.д) - конверсия
        # 2) ARPU - средний чек
        # 3) ARPPU - средний чек среди платящих пользователей
        # 4) Номинативная - распределенно по группам
        # 5) Количественная - непрерывные значения(выручка, количество пользователей и т.д)
        # """)
        number = int(input())
        if number == 'STOP':
            return 'STOP'
        if number == 1:
            return 'CR'
        if number == 2:
            return 'ARPU'
        if number == 3:
            return 'ARPPU'
        if number == 4:
            return 'discrete'
        if number == 5:
            return 'continuous'
        print('Нет, такого варианта. Давай еще раз.')


def get_sigma(metric: str) -> float:
    """
    :param metric: name metric
    :return: standard deviation
    """
    if metric == 'CR' or metric == 'Discrete':
        # print("Введите базовое значение (конверсии)")
        p = float(input())
        return (p*(1-p))**0.5
    else:
        # print("Введите стандартное отклонение(std)")
        std = float(input())
        return std


def effect_size_func(mu_control: float, mu_experiment: float, sigma: float) -> float:
    """
    :param mu_control: mean metric in control group
    :param mu_experiment: mean metric in experiment group
    :param sigma: Standard deviation
    :return: effect_size
    """
    return np.abs(mu_experiment - mu_control) / sigma


def min_sample_size(effect_size: float, power=0.8, alpha=0.05) -> int:
    """
    :param effect_size: from effect_size_func
    :param power: power test
    :param alpha: significant level
    :return: minimal sample size for test
    """
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)

    # find Z_beta from desired power
    z_beta = standard_norm.ppf(power)

    # find Z_alpha
    z_alpha = standard_norm.ppf(1-alpha/2)

    # average of probabilities from both groups

    sample_size = (2 * (z_beta + z_alpha)**2 / effect_size**2)

    return round(sample_size)


def clear_data(df: pd.DataFrame, column_group: str, column_value: str):
    """
    :param df: DataFrame
    :param column_group: name column with group
    :param column_value: name column with value
    :return: clear DataFrame
    """
    message = ''
#   fix types
    if df[column_group].nunique() > 2:
        message = f'Ошибка в записи данных: значений в столбце {column_group} больше 2-х.'
        return pd.DataFrame(), message
    elif df[column_group].nunique() < 2:
        message = f'Ошибка в записи данных: значений в столбце {column_group} меньше 2-х.'
        return pd.DataFrame(), message

    if df[column_value].dtype == 'O':
        try:
            df[column_value] = df[column_value].dropna().apply(lambda x: x.replace(',', '.'))
            df[column_value] = df[column_value].astype('float')
        except:
            message = f'Ошибка в записи данных: значения столбца {column_value} имеют неверный формат.'
            return pd.DataFrame(), message

    elif df[column_value].dtype == 'int64' or df[column_value].dtype == 'int32':
        df[column_value] = df[column_value].astype('float')
    elif df[column_value].dtype != 'float':
        message = f'Ошибка в записи данных: значения столбца {column_value} имеют неверный формат.'
        return pd.DataFrame(), message
    # работа с пропусками
    if df[column_group].isna().sum():
        message += f"Процент пропущенных значений в столбце {column_group} " + \
                  str((df[column_group].isna().sum())/(df.shape[0]) * 100) + "%."
        df.dropna(subset=[column_group], inplace=True)
        message += f"Пропущенные значения были удалены из столбца {column_group}."

    if df[column_value].isna().sum():
        message += f"Процент пропущенных значений в столбце {column_value} " + \
                   str((df[column_value].isna().sum())/(df.shape[0]) * 100) + "%."
        # print(f"""Подскажите, что делать с пропущенными значениями в столбце {column_value}?
        # Введите (без кавычек):\n
        # 'del', если удалить\n
        # 'min', если приравнять к минимальному ({df[column_value].min()})\n
        # 'max', если приравнять к максимальному ({df[column_value].max()})\n
        # 'avg', если приравнять к среднему ({df[column_value].mean()})\n
        # 'median', если приравнять к медиане ({df[column_value].median()})""")
        # <digit>, если заменить на <digit>
        while True:
            str_val = '0'
            if str_val == 'del':
                df[column_value] = df[column_value].dropna()
                message += "\nПропущенные значения были удалены."
                break
            elif str_val == 'min':
                df[column_value].fillna(df[column_value].min(), inplace=True)
                message += "\nПропущенные значения были заменены на минимальное."
                break
            elif str_val == 'max':
                df[column_value].fillna(df[column_value].max(), inplace=True)
                message += "\nПропущенные значения были заменены на максимальное."
                break
            elif str_val == 'avg':
                df[column_value].fillna(df[column_value].mean(), inplace=True)
                message += "\nПропущенные значения были заменены на среднее."
                break
            elif str_val == 'median':
                df[column_value].fillna(df[column_value].median(), inplace=True)
                message += "\nПропущенные значения были заменены на медиану."
                break
            elif str_val.isdigit():
                df[column_value].fillna(float(str_val), inplace=True)
                message += f"\nПропущенные значения были заменены на {str_val}."
                break
            # print('Введено неправильное значение, давай попробуем еще раз.')
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


def get_and_change_outliers(data: pd.DataFrame, name_column_metric: str, n='0'):
    """
    :param data: DataFrame
    :param name_column_metric: name column metric
    :param n: choice for outliers
    :return: DataFrame with fix outliers
    """
    x_1 = data.quantile(0.25)[name_column_metric] - 1.5 * (
            data.quantile(0.75)[name_column_metric] - data.quantile(0.25)[name_column_metric])
    x_2 = data.quantile(0.75)[name_column_metric] + 1.5 * (
                data.quantile(0.75)[name_column_metric] - data.quantile(0.25)[name_column_metric])

    outliers = list(data[(data[name_column_metric] < x_1) | (data[name_column_metric] > x_2)][name_column_metric])
    data_outliers = data[data[name_column_metric].isin(outliers)]
    message = ''
    message += '\nПроцент выбросов ' + str(round(data_outliers.shape[0] / (data.shape[0]) * 100, 2)) + "%."
    if data_outliers.shape[0] > 0:
        # print("""Есть выбросы. Подскажи, что мне с ними сделать?
        # Введите номер варианта:
        # 1) Удалить выбросы;
        # 2) Заменить на максимальные и минимальные значение;
        # 3) Заменить на среднее;
        # 4) Заменить на медиану;
        # 5) Оставить выбросы
        # """)
        max_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].max()
        min_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].min()
        while True:
            if n != '1' and n != '2' and n != '3' and n != '4' and n != '5' and n != 'STOP':
                n = '5'
            if n == 'STOP':
                return pd.DataFrame(), n, message
            if n == '1':
                message += "\nВыбросы были удалены."
                return data[~data[name_column_metric].isin(outliers)], n, message
            if n == '2':
                data[name_column_metric] = data[name_column_metric].\
                    apply(lambda x: min_value if x < min_value else max_value if x > max_value else x)
                message += "\nВыбросы были заменены на максимальное и минимальное значение."
                return data, n, message
            if n == '3':
                mean_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].mean()
                data[name_column_metric] = data[name_column_metric].\
                    apply(lambda x: mean_value if x < min_value else mean_value if x > max_value else x)
                message += "\nВыбросы были заменены на среднее."
                return data, n, message
            if n == '4':
                median_value = data[~data[name_column_metric].isin(outliers)][name_column_metric].median()
                data[name_column_metric] = data[name_column_metric].\
                    apply(lambda x: median_value if x < min_value else median_value if x > max_value else x)
                message += "\nВыбросы были заменены на медиану."
                return data, n, message
            if n == '5':
                return data, n, message
    else:
        return data, '00', message


def get_bootstrap(
    data_column_1,  # числовые значения первой выборки
    data_column_2,  # числовые значения второй выборки
    boot_it=2000,  # количество бутстрап-подвыборок
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


def get_p_value(metric: str, df: pd.DataFrame, name_column_group: str, name_column_metric: str):
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
    if metric == 'CR' or metric == 'Discrete':
        _, _, stats = pg.chi2_independence(df, x=name_column_group, y=name_column_metric)
        p_value = stats.round(3).query('test == "pearson"')['pval'][0]
        power = stats.round(3).query('test == "pearson"')['power'][0]
        return p_value, power, message, df
    if metric == 'ARPU' or metric == 'ARPPU' or metric == 'Continuous':
        if metric == 'ARPPU':
            df_group_1 = df_group_1[df_group_1[name_column_metric] > 0]
            df_group_2 = df_group_2[df_group_2[name_column_metric] > 0]
        distribution = 'normal'
        # print("""Есть выбросы. Подскажи, что мне с ними сделать?
        # Введите номер варианта:
        # 1) Удалить выбросы;
        # 2) Заменить на максимальные и минимальные значение;
        # 3) Заменить на среднее;
        # 4) Заменить на медиану;
        # 5) Оставить выбросы
        # """)
        what_outliers = "5"
        df_group_1, outliers_choice_1, m = get_and_change_outliers(df_group_1, name_column_metric, what_outliers)
        message += f'\nГруппа {name_1}' + m
        df_group_2, outliers_choice_2, m = get_and_change_outliers(df_group_2, name_column_metric, outliers_choice_1)
        message += f'\nГруппа {name_2}' + m
        df = pd.concat([df_group_1, df_group_2])
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
        if distribution == 'normal' and dispersion == 'equal' and outliers_choice_1 != '5' and outliers_choice_2 != '5':
            t_test_result = pg.ttest(df[df[name_column_group] == name_1][name_column_metric],
                                     df[df[name_column_group] == name_2][name_column_metric])
            return round(t_test_result['p-val'][0], 4), round(t_test_result['power'][0], 4), message, df
        if outliers_choice_1 != '5' and outliers_choice_2 != '5':
            return round(get_bootstrap(df_group_1[name_column_metric],
                                       df_group_2[name_column_metric])['p_value'], 4), round(np.nan, 4), message, df
        else:
            p_val = pg.mwu(df_group_1[name_column_metric], df_group_2[name_column_metric],
                           alternative='two-sided')['p-val'][0]
            return round(p_val, 4), round(np.nan, 4), message, df


def get_conclusion(df: pd.DataFrame, name_column_group: str, name_column_metric: str, p_val: float):
    """
    :param df: DataFrame
    :param name_column_group: name column group
    :param name_column_metric: name column metric
    :param p_val: p_value
    """
    group_names = list(df[name_column_group].unique())
    name_1 = group_names[0]
    name_2 = group_names[1]
    df_group_1 = df[df[name_column_group] == name_1]
    df_group_2 = df[df[name_column_group] == name_2]
    mean_group_1 = df_group_1[name_column_metric].mean()
    mean_group_2 = df_group_2[name_column_metric].mean()
    if p_val < 0.05:
        return f'''Среднее в группе {name_1}: {mean_group_1}.\nСреднее в группе {name_2}: {mean_group_2}.\nРазличия в средних статистически значимы.'''
    else:
        return f'''Среднее в группе {name_1}: {mean_group_1}.\nСреднее в группе {name_2}: {mean_group_2}.\nРазличия в средних статистически незначимы.'''

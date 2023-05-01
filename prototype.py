import logging
import library_for_ab
from telegram import __version__ as TG_VER
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import re
import numpy as np
import os

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 5):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


ALPHA_SS, BETA_SS, STD_SS, MEAN_DATA_SS, PATH_SS, DELTA_SS = '', '', '', '', '', ''
ALPHA_SSR, BETA_SSR, PATH_OR_STD_SSR, STD_SSR, MEAN_SSR, PATH_SSR, COLUMN_NAME_SSR, EFFECT_SSR, RATIO_SSR = range(9)
METRIC, PATH, GROUP_COLUMN, VALUES_COLUMN, DECISION_MISS, CHANGE_OUTLIERS, GET_P_VALUE = range(7)
METRIC_M, GROUP_COLUMN_M, VALUES_COLUMN_M, DATAFRAME_M = '',  '', '', pd.DataFrame()
DESC_METRIC_VALUE_COLUMNS = []
IS_OUTLIERS_CONT_METRIC = []
CONT_METRIC_VALUE_COLUMNS = []
RANK_METRIC_VALUE_COLUMNS = []


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start."""
    await update.message.reply_text(
        "Привет! Я помогу тебе с проведением A/B теста.\n"
        "Я могу помочь тебе с определением количество наблюдений в выборке. Для этого введите команду /sample_size\n\n"
        "Еще я могу тебе помочь с анализом A/B теста. Для этого введите команду /ab_test\n\n"
        "Также я могу помочь с анализом A/A теста. Для этого введите команду /aa_test\n\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
    )
    return ConversationHandler.END


async def sample_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start."""
    await update.message.reply_text(
        "Ты выбрал блок с определение размера выборки для проведения A/B теста.\n\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
        "Для того, чтобы я тебе помог, мне необходима дополнительная информация.\n\n"
        "Какой уровень значимости(alpha) мы будем использовать?\n"
        "Если не знаешь, то будем использовать значение по-умолчанию(0.05), введи 0.05.\n",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ALPHA_SSR

async def alpha_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global ALPHA_SS
    try:
        ALPHA_SS = float(update.message.text)
    except:
        raise ValueError("Alpha cant convert to float")
    if ALPHA_SS >= 1 or ALPHA_SS <= 0:
        raise ValueError("Incorrect alpha")
    user = update.message.from_user
    logger.info("alpha of %s: %s", user.first_name, ALPHA_SS)
    
    await update.message.reply_text(
        "Отлично!.\n\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
        "Какая мощность(beta) теста нас интересует?.\n"
        "Если не знаешь, то будем использовать значение по-умолчанию(0.8), введи 0.8.\n",
        reply_markup=ReplyKeyboardRemove(),
    )
    return BETA_SSR

async def beta_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global BETA_SS
    try:
        BETA_SS = float(update.message.text)
    except:
        raise ValueError("Beta cant convert to float")
    if BETA_SS >= 1 or BETA_SS <= 0:
        raise ValueError("Incorrect beta")
    user = update.message.from_user
    logger.info("beta of %s: %s", user.first_name, BETA_SS)
    
    reply_keyboard = [['Знаю стандартное отклонение в данных', 'Помочь определить']]
    await update.message.reply_text(
        "Отлично!.\n\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
        "Далее необходимо определить стандартное отклонение в данных.\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Metric?"
        ),
    )
    return PATH_OR_STD_SSR

async def path_or_std_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    path_or_std = update.message.text
    user = update.message.from_user
    logger.info("std of %s: %s", user.first_name, path_or_std)
    if path_or_std == 'Знаю стандартное отклонение в данных':
        await update.message.reply_text(
        "Введите стандартное отклонение.",
        reply_markup=ReplyKeyboardRemove(),
    )
        return STD_SSR
    await update.message.reply_text(
        "Пожалуйста, пришлите мне ссылку на файл на гугл диске.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PATH_SSR

async def std_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global STD_SS
    try:
        STD_SS = float(update.message.text)
        user = update.message.from_user
        logger.info("std of %s: %s", user.first_name, STD_SS)
    except:
        raise ValueError("STD cant convert to float")
    await update.message.reply_text(
        "Введите среднее в выборке.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return MEAN_SSR

async def mean_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global MEAN_DATA_SS
    try:
        MEAN_DATA_SS = float(update.message.text)
        user = update.message.from_user
        logger.info("std of %s: %s", user.first_name, MEAN_DATA_SS)
    except:
        raise ValueError("MEAN cant convert to float")
    await update.message.reply_text(
            "Отлично!.\n\n"
            "Отправь /cancel чтобы прекратить общение со мной.\n\n"
            f"Среднее в выборке {MEAN_DATA_SS}\n"
            "Введи ожидаемый эффект(я бы хотел обнаружить, что среднее станет таким-то значением).\n",
            reply_markup=ReplyKeyboardRemove(),)
    return EFFECT_SSR

async def path_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global PATH_SS
    path = update.message.text
    PATH_SS = 'https://drive.google.com/uc?id=' + path.split('/')[-2]
    await update.message.reply_text(
    "Отлично!.\n\n"
    "Отправь /cancel чтобы прекратить общение со мной.\n\n"
    "Введи название колонки с данными.\n",
    reply_markup=ReplyKeyboardRemove(),)
    return COLUMN_NAME_SSR

async def columns_name_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global STD_SS
    global MEAN_DATA_SS
    column_name = update.message.text
    df = pd.read_csv(PATH_SS)
    try:
        STD_SS = df[column_name].std()
        user = update.message.from_user
        logger.info("std of %s: %s", user.first_name, STD_SS)
        MEAN_DATA_SS = df[column_name].mean()
    except:
        raise ValueError("Incorrect data")
    await update.message.reply_text(
            "Отлично!.\n\n"
            "Отправь /cancel чтобы прекратить общение со мной.\n\n"
            f"Среднее в выборке {MEAN_DATA_SS}\n"
            "Введи ожидаемый эффект(я бы хотел обнаружить, что среднее станет таким-то значением).\n",
            reply_markup=ReplyKeyboardRemove(),)
    return EFFECT_SSR
    
async def effect_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global DELTA_SS
    try:
        mu_control = float(update.message.text)
    except:
        raise ValueError("mu_control cant convert to float")
    user = update.message.from_user
    logger.info("mu_control of %s: %s", user.first_name, mu_control)
    
    DELTA_SS = np.abs(MEAN_DATA_SS - mu_control)
    
    await update.message.reply_text(
            "Отлично!.\n\n"
            "Отправь /cancel чтобы прекратить общение со мной.\n\n"
            "Введи сколько процентов будет в тестовой группе.\n",
            reply_markup=ReplyKeyboardRemove(),)
    return RATIO_SSR

async def ratio_ss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        test_pers = float(update.message.text)
    except:
        raise ValueError("test_pers cant convert to float")
    user = update.message.from_user
    logger.info("mu_control of %s: %s", user.first_name, test_pers)
    
    if test_pers <=0 or test_pers>=100:
        raise ValueError("test_pers incorrect")
    ratio = (100-test_pers)/test_pers
    n1, n2 = library_for_ab.calculate_sample_size(DELTA_SS, STD_SS, alpha=ALPHA_SS, beta=BETA_SS,  ratio=ratio)
    await update.message.reply_text(
            f"Размер тестовой группы {n2}, размер контрольной группы {n1}.\n",
            reply_markup=ReplyKeyboardRemove(),)
    
    return ConversationHandler.END

async def aa_test_aa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start."""
    reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
    await update.message.reply_text(
        "Привет! Я помогу тебе с A/A тестом. Пожалуйста, пришлите мне ссылку на файл на гугл диске.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PATH

async def ab_test_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start."""
    reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
    await update.message.reply_text(
        "Привет! Я помогу тебе с A/B тестом. Пожалуйста, пришлите мне ссылку на файл на гугл диске.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PATH



async def path_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global DATAFRAME_M
    """Stores the selected path and asks for a group column."""
    user = update.message.from_user
    url = update.message.text
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    DATAFRAME_M = pd.read_csv(url)
    
    logger.info("Path of %s: %s", user.first_name, url)
    await update.message.reply_text(
        "Пожалуйста, напишите мне название колонки с группами.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return GROUP_COLUMN


async def group_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the group column and asks for value column."""
    global DATAFRAME_M
    global GROUP_COLUMN_M
    user = update.message.from_user
    GROUP_COLUMN_M = update.message.text
    logger.info("Group column of %s: %s", user.first_name, GROUP_COLUMN_M)
        
    if GROUP_COLUMN_M not in DATAFRAME_M.columns:
        await update.message.reply_text('Неверное название колонки с группами. Необходимо начать сначала.')
        return ConversationHandler.END
    DATAFRAME_M, message = library_for_ab.clear_data_group_columns(DATAFRAME_M, GROUP_COLUMN_M)
    
    reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
    await update.message.reply_text(message + 
        "\n\nСкажи мне, какие метрики мы будем отслеживать.\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
        "Выбери тип данных и перечисли все названия колонок в датасете по одному за раз, которые относятся к этому типу данных\n\n"
        "Discrete - данные, которые можно разбить на классы(пол, цвет глаз и т.д)\n\n"
        "Continuous - непрерывные данные(рост, вес и т.д)\n\n"
        "Ranking - ранговые данные(место в соревновании)",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        ),
    )
    return METRIC


async def metric_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected metric and asks for a path."""
    global METRIC_M
    user = update.message.from_user
    METRIC_M = update.message.text
    logger.info("Metric of %s: %s", user.first_name, METRIC_M)
    await update.message.reply_text(
        f"Пожалуйста, напиши, название колонки, в которых данные типа {METRIC_M}.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return VALUES_COLUMN


async def value_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global VALUES_COLUMN_M
    global DATAFRAME_M
    is_outliers = False
    user = update.message.from_user
    VALUES_COLUMN_M = update.message.text
    logger.info("value_columns of %s: %s", user.first_name, VALUES_COLUMN_M)
    if VALUES_COLUMN_M not in DATAFRAME_M.columns:
        await update.message.reply_text('Неверное название колонки со значениями. Необходимо начать сначала.')
        return ConversationHandler.END
    
    if METRIC_M == "Discrete":
        DESC_METRIC_VALUE_COLUMNS.append(VALUES_COLUMN_M)
    elif METRIC_M == "Continuous":
        CONT_METRIC_VALUE_COLUMNS.append(VALUES_COLUMN_M)
    elif METRIC_M == "Ranking":
        RANK_METRIC_VALUE_COLUMNS.append(VALUES_COLUMN_M)
        
    DATAFRAME_M, message, have_missing = library_for_ab.clear_data(DATAFRAME_M, GROUP_COLUMN_M, VALUES_COLUMN_M)
    if DATAFRAME_M.shape[0] == 0:
        await update.message.reply_text(message + ' Необходимо начать сначала.')
        return ConversationHandler.END
    if have_missing:
        reply_keyboard = [[ "Удалить", "Минимальное", 'Максимальное', 'Среднее', 'Медиану']]
        await update.message.reply_text(message + " \n\nЧто делать с пропущенными значениями? Выбери из предложенного или введи число, на которое заменим пропущенные значения:\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
        return DECISION_MISS
    
    if METRIC_M == "Continuous":
        message = ''
        for group in DATAFRAME_M[GROUP_COLUMN_M].unique():
            message_outliers, flag_outliers = library_for_ab.get_outliers(DATAFRAME_M[DATAFRAME_M[GROUP_COLUMN_M] == group], VALUES_COLUMN_M)
            message += f'В группе {group}: ' + message_outliers + '\n'
            is_outliers += flag_outliers
        if is_outliers:
            reply_keyboard = [[ "Удалить", "Максимальное и минимальное", 'Среднее', 'Медиану', 'Оставить']]
            await update.message.reply_text(message_miss + "\n\n" + message + " \n\nЕсть выбросы. Подскажи, что мне с ними сделать? Выбери из предложенного:\n",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True),)
            return CHANGE_OUTLIERS
        IS_OUTLIERS_CONT_METRIC.append(is_outliers)
        
    reply_keyboard = [[ "Да", "Нет"]]
    await update.message.reply_text(message+ " \n\nХочешь ли ты еще добавить данные для сравнения?\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
    return GET_P_VALUE
    
    
    
async def decision_missing_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global DATAFRAME_M
    is_outliers = False
    user = update.message.from_user
    decision_miss_value = update.message.text
    logger.info("decision_miss_value of %s: %s", user.first_name, decision_miss_value)
    DATAFRAME_M, message_miss = library_for_ab.missing_values(DATAFRAME_M, GROUP_COLUMN_M, VALUES_COLUMN_M, str_val= decision_miss_value)
    message = ''
    if METRIC_M == "Continuous":
        message = ''
        for group in DATAFRAME_M[GROUP_COLUMN_M].unique():
            message_outliers, flag_outliers = library_for_ab.get_outliers(DATAFRAME_M[DATAFRAME_M[GROUP_COLUMN_M] == group], VALUES_COLUMN_M)
            message += f'В группе {group}: ' + message_outliers + '\n'
            is_outliers += flag_outliers
        if is_outliers:
            reply_keyboard = [[ "Удалить", "Максимальное и минимальное", 'Среднее', 'Медиану', 'Оставить']]
            await update.message.reply_text(message_miss + "\n\n" + message + " \n\nЕсть выбросы. Подскажи, что мне с ними сделать? Выбери из предложенного:\n",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True),)
            return CHANGE_OUTLIERS
        IS_OUTLIERS_CONT_METRIC.append(is_outliers)
        
    reply_keyboard = [[ "Да", "Нет"]]
    await update.message.reply_text(message+ " \n\nХочешь ли ты еще добавить данные для сравнения?\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
    return GET_P_VALUE
    
    
async def change_outliers_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global DATAFRAME_M
    user = update.message.from_user
    decision = update.message.text
    DATAFRAME_M, message, is_outliers = library_for_ab.change_outliers(DATAFRAME_M, VALUES_COLUMN_M, decision)
    IS_OUTLIERS_CONT_METRIC.append(is_outliers)
    reply_keyboard = [[ "Да", "Нет"]]
    await update.message.reply_text(message+ " \n\nХочешь ли ты еще добавить данные для сравнения?\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
    return GET_P_VALUE
    

async def p_value_aa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global CONT_METRIC_VALUE_COLUMNS
    global DESC_METRIC_VALUE_COLUMNS
    global RANK_METRIC_VALUE_COLUMNS
    
    user = update.message.from_user
    decision_more_metric = update.message.text
    
    if True in IS_OUTLIERS_CONT_METRIC:
        await update.message.reply_text(
            "К сожалению, я не могу провести A/A тест, так как в данных есть выбросы, давай начнем с начала и избавимся от выбросов.")
        return ConversationHandler.END
    
    
    
    if decision_more_metric == 'Да':
        reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
        await update.message.reply_text(
            "Выбери тип данных:\n\n"
            "Discrete - данные, которые можно разбить на классы(пол, цвет глаз и т.д)\n\n"
            "Continuous - непрерывные данные(рост, вес и т.д)\n\n"
            "Ranking - ранговые данные(место в соревновании)",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True,
            ),
        )
        return METRIC
    
    if CONT_METRIC_VALUE_COLUMNS != []:
        final_dataset_cont = DATAFRAME_M.groupby(GROUP_COLUMN_M, as_index=0)[CONT_METRIC_VALUE_COLUMNS].mean()
        dict_row = {GROUP_COLUMN_M:['diff', 'p_val']}
        for i in range(len(CONT_METRIC_VALUE_COLUMNS)):
            value_columns = CONT_METRIC_VALUE_COLUMNS[i]
            is_outliers = IS_OUTLIERS_CONT_METRIC[i]
            p_value, _, _ = library_for_ab.get_p_value("Continuous", DATAFRAME_M, GROUP_COLUMN_M, value_columns, is_outliers)
            dict_row[value_columns] = [round((final_dataset_cont[value_columns][1]/final_dataset_cont[value_columns][0]-1)*100, 2), round(p_value, 3)]
        final_dataset_cont = final_dataset_cont.append(pd.DataFrame(dict_row))
    
    if DESC_METRIC_VALUE_COLUMNS != []:
        final_datasets_desc_list = []
        for i in range(len(DESC_METRIC_VALUE_COLUMNS)):
            value_columns = DESC_METRIC_VALUE_COLUMNS[i]
            final_datasets_desc = DATAFRAME_M.pivot_table(index=GROUP_COLUMN_M, columns=value_columns, values='count')
            final_datasets_desc = (final_datasets_desc.T / final_datasets_desc.T.sum()).T
            p_value, _, _ = library_for_ab.get_p_value("Discrete", DATAFRAME_M, GROUP_COLUMN_M, value_columns, False)
            dict_desc_diff = {GROUP_COLUMN_M:['diff', 'p_val']}
            for desc_obj in final_datasets_desc.columns:
                dict_desc_diff[desc_obj] = [round((final_datasets_desc[desc_obj][1]/final_datasets_desc[desc_obj][0]-1)*100, 2), round(p_value, 3)]
            final_datasets_desc = final_datasets_desc.reset_index()
            final_datasets_desc = final_datasets_desc.append(pd.DataFrame(dict_desc_diff))
        final_datasets_desc_list.append(final_datasets_desc)
    
    if RANK_METRIC_VALUE_COLUMNS != []:
        final_dataset_rank = DATAFRAME_M.groupby(GROUP_COLUMN_M, as_index=0)[RANK_METRIC_VALUE_COLUMNS].mean()
        dict_p_val = {GROUP_COLUMN_M:['p_val']}
        for i in range(len(RANK_METRIC_VALUE_COLUMNS)):
            value_columns = RANK_METRIC_VALUE_COLUMNS[i]
            p_value, _, _ = library_for_ab.get_p_value("Ranking", DATAFRAME_M, GROUP_COLUMN_M, value_columns, False)
            dict_p_val[value_columns] = [p_value]
        final_dataset_rank = final_dataset_rank.append(pd.DataFrame(dict_p_val))
    
    path = 'result/result_' + str(user.id) + '.xlsx'
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        if CONT_METRIC_VALUE_COLUMNS != []:
            final_dataset_cont.to_excel(writer, sheet_name='Сводная таблица', startrow=1, startcol=1, index=False)
            
        if RANK_METRIC_VALUE_COLUMNS != []:
            final_dataset_rank.to_excel(writer, sheet_name='Сводная таблица', startrow=6, startcol=1, index=False)
        
        if DESC_METRIC_VALUE_COLUMNS != []:
            for i in range(len(final_datasets_desc_list)):
                df = final_datasets_desc_list[i]
                df.to_excel(writer, sheet_name='Сводная таблица', startrow=11 + i*5, startcol=1, index=False)
        list_image_cont_hist, list_image_cont_kde, list_image_cont_qq, list_image_cont_boxplot, list_image_desc = library_for_ab.get_image(DATAFRAME_M, GROUP_COLUMN_M, CONT_METRIC_VALUE_COLUMNS, DESC_METRIC_VALUE_COLUMNS)
        list_image_cont_aa, list_image_desc_aa, list_image_rank_aa = library_for_ab.get_image_aa(DATAFRAME_M, GROUP_COLUMN_M, CONT_METRIC_VALUE_COLUMNS, DESC_METRIC_VALUE_COLUMNS, RANK_METRIC_VALUE_COLUMNS)
        
        pd.DataFrame().to_excel(writer, sheet_name='Визуализация', startrow=1, startcol=1, index=False)
        worksheet = writer.sheets['Визуализация']
        line = 1
        for i in range(len(list_image_cont_hist)):
            list_image_cont_aa[i].savefig(f'image/image_cont_aa_'+ str(line) + '_'+ str(user.id) +'.png')
            list_image_cont_hist[i].savefig(f'image/image_hist_'+ str(line) + '_'+ str(user.id) +'.png')
            list_image_cont_kde[i].savefig(f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_cont_qq[i].savefig(f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_cont_boxplot[i].savefig(f'image/image_boxplot_' + str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('A'+str(line), f'image/image_cont_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('H'+str(line), f'image/image_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('O'+str(line), f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('V'+str(line), f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('AC'+str(line), f'image/image_boxplot_'+ str(line)+ '_'+ str(user.id) +'.png')
            line += 20
        for i in range(len(list_image_desc)):
            line += 20
            list_image_desc_aa[i].savefig(f'image/image_desc_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_desc[i].savefig(f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('A'+str(line), f'image/image_desc_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('H'+str(line), f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
        for i in range(len(list_image_rank_aa)):
            line += 20
            list_image_rank_aa[i].savefig(f'image/image_rank_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('A'+str(line), f'image/image_rank_aa_'+ str(line)+ '_'+ str(user.id) +'.png')

            
    line = 1
    for i in range(len(list_image_cont_hist)):
        os.remove(f'image/image_cont_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_boxplot_'+ str(line)+ '_'+ str(user.id) +'.png')
        line += 20
    for i in range(len(list_image_desc)):
        line += 20
        os.remove(f'image/image_desc_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
    for i in range(len(list_image_rank_aa)):
        line += 20
        os.remove(f'image/image_rank_aa_'+ str(line)+ '_'+ str(user.id) +'.png')
    
    await update.message.reply_document(
        document=open(path, "rb"),
        filename="result.xlsx"
    )
    os.remove(path)
    RANK_METRIC_VALUE_COLUMNS = []
    DESC_METRIC_VALUE_COLUMNS = []
    CONT_METRIC_VALUE_COLUMNS = []
    list_image_cont_hist, list_image_cont_kde, list_image_cont_qq, list_image_cont_boxplot, list_image_desc, list_image_cont_aa, list_image_desc_aa, list_image_rank_aa = [], [], [], [], [], [], [], []
    try:
        await update.message.reply_text('Все получилось!')
    except:
        pass
    return ConversationHandler.END

    
    
async def p_value_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global CONT_METRIC_VALUE_COLUMNS
    global DESC_METRIC_VALUE_COLUMNS
    global RANK_METRIC_VALUE_COLUMNS

    user = update.message.from_user
    decision_more_metric = update.message.text
    if decision_more_metric == 'Да':
        reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
        await update.message.reply_text(
            "Выбери тип данных:\n\n"
            "Discrete - данные, которые можно разбить на классы(пол, цвет глаз и т.д)\n\n"
            "Continuous - непрерывные данные(рост, вес и т.д)\n\n"
            "Ranking - ранговые данные(место в соревновании)",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True,
            ),
        )
        return METRIC
    
    if CONT_METRIC_VALUE_COLUMNS != []:
        final_dataset_cont = DATAFRAME_M.groupby(GROUP_COLUMN_M, as_index=0)[CONT_METRIC_VALUE_COLUMNS].mean()
        dict_row = {GROUP_COLUMN_M:['diff', 'p_val']}
        for i in range(len(CONT_METRIC_VALUE_COLUMNS)):
            value_columns = CONT_METRIC_VALUE_COLUMNS[i]
            is_outliers = IS_OUTLIERS_CONT_METRIC[i]
            p_value, _, _ = library_for_ab.get_p_value("Continuous", DATAFRAME_M, GROUP_COLUMN_M, value_columns, is_outliers)
            dict_row[value_columns] = [round((final_dataset_cont[value_columns][1]/final_dataset_cont[value_columns][0]-1)*100, 2), round(p_value, 3)]
        final_dataset_cont = final_dataset_cont.append(pd.DataFrame(dict_row))
    
    if DESC_METRIC_VALUE_COLUMNS != []:
        final_datasets_desc_list = []
        for i in range(len(DESC_METRIC_VALUE_COLUMNS)):
            value_columns = DESC_METRIC_VALUE_COLUMNS[i]
            final_datasets_desc = DATAFRAME_M.pivot_table(index=GROUP_COLUMN_M, columns=value_columns, values='count')
            final_datasets_desc = (final_datasets_desc.T / final_datasets_desc.T.sum()).T
            p_value, _, _ = library_for_ab.get_p_value("Discrete", DATAFRAME_M, GROUP_COLUMN_M, value_columns, False)
            dict_desc_diff = {GROUP_COLUMN_M:['diff', 'p_val']}
            for desc_obj in final_datasets_desc.columns:
                dict_desc_diff[desc_obj] = [round((final_datasets_desc[desc_obj][1]/final_datasets_desc[desc_obj][0]-1)*100, 2), round(p_value, 3)]
            final_datasets_desc = final_datasets_desc.reset_index()
            final_datasets_desc = final_datasets_desc.append(pd.DataFrame(dict_desc_diff))
        final_datasets_desc_list.append(final_datasets_desc)
    
    if RANK_METRIC_VALUE_COLUMNS != []:
        final_dataset_rank = DATAFRAME_M.groupby(GROUP_COLUMN_M, as_index=0)[RANK_METRIC_VALUE_COLUMNS].mean()
        dict_p_val = {GROUP_COLUMN_M:['p_val']}
        for i in range(len(RANK_METRIC_VALUE_COLUMNS)):
            value_columns = RANK_METRIC_VALUE_COLUMNS[i]
            p_value, _, _ = library_for_ab.get_p_value("Ranking", DATAFRAME_M, GROUP_COLUMN_M, value_columns, False)
            dict_p_val[value_columns] = [p_value]
        final_dataset_rank = final_dataset_rank.append(pd.DataFrame(dict_p_val))
    
    path = 'result/result_' + str(user.id) + '.xlsx'
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        if CONT_METRIC_VALUE_COLUMNS != []:
            final_dataset_cont.to_excel(writer, sheet_name='Сводная таблица', startrow=1, startcol=1, index=False)
            
        if RANK_METRIC_VALUE_COLUMNS != []:
            final_dataset_rank.to_excel(writer, sheet_name='Сводная таблица', startrow=6, startcol=1, index=False)
        
        if DESC_METRIC_VALUE_COLUMNS != []:
            for i in range(len(final_datasets_desc_list)):
                df = final_datasets_desc_list[i]
                df.to_excel(writer, sheet_name='Сводная таблица', startrow=11 + i*5, startcol=1, index=False)
        list_image_cont_hist, list_image_cont_kde, list_image_cont_qq, list_image_cont_boxplot, list_image_desc = library_for_ab.get_image(DATAFRAME_M, GROUP_COLUMN_M, CONT_METRIC_VALUE_COLUMNS, DESC_METRIC_VALUE_COLUMNS)
        
        pd.DataFrame().to_excel(writer, sheet_name='Визуализация', startrow=1, startcol=1, index=False)
        worksheet = writer.sheets['Визуализация']
        line = 1
        for i in range(len(list_image_cont_hist)):
            list_image_cont_hist[i].savefig(f'image/image_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_cont_kde[i].savefig(f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_cont_qq[i].savefig(f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
            list_image_cont_boxplot[i].savefig(f'image/image_boxplot_' + str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('A'+str(line), f'image/image_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('H'+str(line), f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('O'+str(line), f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('V'+str(line), f'image/image_boxplot_'+ str(line)+ '_'+ str(user.id) +'.png')
            line += 20
        for i in range(len(list_image_desc)):
            line += 20
            list_image_desc[i].savefig(f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
            worksheet.insert_image('A'+str(line), f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
    line = 1
    for i in range(len(list_image_cont_hist)):
        os.remove(f'image/image_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_kde_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_qq_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_boxplot_'+ str(line)+ '_'+ str(user.id) +'.png')
        line += 20
    for i in range(len(list_image_desc)):
        os.remove(f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
        os.remove(f'image/image_desc_hist_'+ str(line)+ '_'+ str(user.id) +'.png')
    
    await update.message.reply_document(
        document=open(path, "rb"),
        filename="result.xlsx"
    )
    os.remove(path)
    RANK_METRIC_VALUE_COLUMNS = []
    DESC_METRIC_VALUE_COLUMNS = []
    CONT_METRIC_VALUE_COLUMNS = []
    list_image_cont_hist, list_image_cont_kde, list_image_cont_qq, list_image_cont_boxplot, list_image_desc = [], [], [], [], []
    try:
        await update.message.reply_text('Все получилось!')
    except:
        pass
    return ConversationHandler.END





    


# async def ab_test_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Start."""
#     reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
#     await update.message.reply_text(
#         "Привет! Я помогу тебе с A/B тестом. Скажи мне, какие метрики мы будем отслеживать.\n"
#         "Отправь /cancel чтобы прекратить общение со мной.\n\n"
#         "Выбери тип данных и перечисли все названия колонок в датасете через запятую, которые относятся к этому типу данных\n\n"
#         "Discrete - данные, которые можно разбить на классы(пол, цвет глаз и т.д)\n\n"
#         "Continuous - непрерывные данные(рост, вес и т.д)\n\n"
#         "Ranking - ранговые данные(место в соревновании)",
#         reply_markup=ReplyKeyboardMarkup(
#             reply_keyboard, one_time_keyboard=True, input_field_placeholder="Metric?"
#         ),
#     )
#     return METRIC



# async def metric_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Stores the selected metric and asks for a path."""
#     global METRIC_M
#     user = update.message.from_user
#     METRIC_M = update.message.text
#     logger.info("Metric of %s: %s", user.first_name, METRIC_M)
#     await update.message.reply_text(
#         "Пожалуйста, пришлите мне ссылку на файл на гугл диске.",
#         reply_markup=ReplyKeyboardRemove(),
#     )
#     return PATH


# async def path_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Stores the selected path and asks for a group column."""
#     global PATH_M
#     user = update.message.from_user
#     PATH_M = update.message.text
#     logger.info("Path of %s: %s", user.first_name, PATH_M)
#     await update.message.reply_text(
#         "Пожалуйста, напишите мне название колонки с группами.",
#         reply_markup=ReplyKeyboardRemove(),
#     )
#     return GROUP_COLUMN


# async def group_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Stores the group column and asks for value column."""
#     global GROUP_COLUMN_M
#     user = update.message.from_user
#     GROUP_COLUMN_M = update.message.text
#     logger.info("Group column of %s: %s", user.first_name, GROUP_COLUMN_M)
#     await update.message.reply_text(
#         "Пожалуйста, напишите мне название колонки со значениями.",
#         reply_markup=ReplyKeyboardRemove(),
#     )
#     return VALUES_COLUMN


# async def value_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Stores the info about value column."""
#     global VALUES_COLUMN_M
#     global DATAFRAME
#     user = update.message.from_user
#     VALUES_COLUMN_M = update.message.text
#     logger.info("Value column of %s: %s", user.first_name, VALUES_COLUMN_M)

#     url = PATH_M
#     url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
#     df = pd.read_csv(url)
#     if GROUP_COLUMN_M not in df.columns:
#         await update.message.reply_text('Неверное название колонки с группами. Необходимо начать сначала.')
#         return ConversationHandler.END
#     if VALUES_COLUMN_M not in df.columns:
#         await update.message.reply_text('Неверное название колонки со значениями. Необходимо начать сначала.')
#         return ConversationHandler.END
#     DATAFRAME, message, have_missing = library_for_ab.clear_data(df, GROUP_COLUMN_M, VALUES_COLUMN_M)
#     if df.shape[0] == 0:
#         await update.message.reply_text(message + ' Необходимо начать сначала.')
#         return ConversationHandler.END
#     if have_missing:
#         reply_keyboard = [[ "Удалить", "Минимальное", 'Максимальное', 'Среднее', 'Медиану']]
#         await update.message.reply_text(message + " \n\nЧто делать с пропущенными значениями? Выбери из предложенного или введи число, на которое заменим пропущенные значения:\n",
#         reply_markup=ReplyKeyboardMarkup(
#             reply_keyboard, one_time_keyboard=True),)
#         return DECISION_MISS
#     p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
#     try:
#         await update.message.reply_text(message)
#     except:
#         pass
#     final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
#     if p_value < 0.05:
#         await update.message.reply_text(message+'\n'
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     else:
#         await update.message.reply_text(
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     return ConversationHandler.END

    
# async def decision_missing_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     global IS_OUTLIERS
#     global DATAFRAME
#     IS_OUTLIERS = False
#     user = update.message.from_user
#     decision_miss_value = update.message.text
#     logger.info("decision_miss_value of %s: %s", user.first_name, decision_miss_value)
#     DATAFRAME, message_miss = library_for_ab.missing_values(DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, decision_miss_value)
#     dict_outliers = {}
#     message = ''
#     for group in DATAFRAME[GROUP_COLUMN_M].unique():
#         message_outliers, flag_outliers = library_for_ab.get_outliers(DATAFRAME[DATAFRAME[GROUP_COLUMN_M] == group], VALUES_COLUMN_M)
#         message += f'В группе {group}: ' + message_outliers + '\n'
#         IS_OUTLIERS += flag_OUTLIERS
#     if IS_OUTLIERS:
#         reply_keyboard = [[ "Удалить", "Максимальное и минимальное", 'Среднее', 'Медиану', 'Оставить']]
#         await update.message.reply_text(message_miss + "\n\n" + message + " \n\nЕсть выбросы. Подскажи, что мне с ними сделать? Выбери из предложенного:\n",
#         reply_markup=ReplyKeyboardMarkup(
#             reply_keyboard, one_time_keyboard=True),)
#         return CHANGE_OUTLIERS
#     p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
#     try:
#         await update.message.reply_text(message)
#     except:
#         pass
#     final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
#     if p_value < 0.05:
#         await update.message.reply_text(message+'\n'
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     else:
#         await update.message.reply_text(
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     return ConversationHandler.END


# async def change_outliers_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     global IS_OUTLIERS
#     global DATAFRAME
#     user = update.message.from_user
#     decision = update.message.text
#     DATAFRAME, message, IS_OUTLIERS = library_for_ab.change_outliers(DATAFRAME[DATAFRAME[GROUP_COLUMN_M] == group], VALUES_COLUMN_M, decision)
#     await update.message.reply_text(message)
#     p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
#     try:
#         await update.message.reply_text(message)
#     except:
#         pass
#     final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
#     if p_value < 0.05:
#         await update.message.reply_text(message+'\n'
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     else:
#         await update.message.reply_text(
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     return ConversationHandler.END

# async def p_value_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
#     try:
#         await update.message.reply_text(message)
#     except:
#         pass
#     final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
#     if p_value < 0.05:
#         await update.message.reply_text(message+'\n'
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     else:
#         await update.message.reply_text(
#             'p_value: ' + str(p_value) + ' мощность: ' + str(power))
#         await update.message.reply_text(final_message)
#     return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "Пока!", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("5871285096:AAFQqR0yppdxsW0tn29liLpwihrUHLFtXBw").read_timeout(30).write_timeout(30).connect_timeout(30).build()
    
    conv_handler_start = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={},
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    
    conv_handler_sample_size = ConversationHandler(
        entry_points=[CommandHandler("sample_size", sample_size)],
        states={
            ALPHA_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, alpha_ss)],
            BETA_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, beta_ss)],
            PATH_OR_STD_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, path_or_std_ss)],
            STD_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, std_ss)],
            MEAN_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, mean_ss)],
            PATH_SSR: [MessageHandler(filters.Regex("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"), path_ss)],
            COLUMN_NAME_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, columns_name_ss)],
            EFFECT_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, effect_ss)],
            RATIO_SSR: [MessageHandler(filters.TEXT & ~filters.COMMAND, ratio_ss)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    conv_handler_ab_test = ConversationHandler(
        entry_points=[CommandHandler("ab_test", ab_test_ab)],
        states={
            METRIC: [MessageHandler(filters.Regex("^(Discrete|Continuous|Ranking)$"), metric_ab)],
            PATH: [MessageHandler(filters.Regex("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"), path_ab)],
            GROUP_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, group_column_ab)],
            VALUES_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, value_column_ab)],
            DECISION_MISS: [MessageHandler(filters.TEXT & ~filters.COMMAND, decision_missing_ab)],
            CHANGE_OUTLIERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, change_outliers_ab)],
            GET_P_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, p_value_ab)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    conv_handler_ab_test = ConversationHandler(
        entry_points=[CommandHandler("aa_test", aa_test_aa)],
        states={
            METRIC: [MessageHandler(filters.Regex("^(Discrete|Continuous|Ranking)$"), metric_ab)],
            PATH: [MessageHandler(filters.Regex("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"), path_ab)],
            GROUP_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, group_column_ab)],
            VALUES_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, value_column_ab)],
            DECISION_MISS: [MessageHandler(filters.TEXT & ~filters.COMMAND, decision_missing_ab)],
            CHANGE_OUTLIERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, change_outliers_ab)],
            GET_P_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, p_value_aa)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler_start)
    application.add_handler(conv_handler_sample_size)
    application.add_handler(conv_handler_ab_test)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
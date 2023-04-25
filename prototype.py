import logging
import library_for_ab
from telegram import __version__ as TG_VER
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import re
import numpy as np

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
METRIC_M, PATH_M, GROUP_COLUMN_M, VALUES_COLUMN_M, DATAFRAME, IS_OUTLIERS = '', '', '', '', pd.DataFrame(), False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start."""
    await update.message.reply_text(
        "Привет! Я помогу тебе с проведением A/B теста.\n"
        "Я могу помочь тебе с определением количество наблюдений в выборке. Для этого введите команду /sample_size\n\n"
        "Еще я могу тебе помочь с анализом A/B теста. Для этого введите команду /ab_test\n\n"
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

async def ab_test_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start."""
    reply_keyboard = [[ "Discrete", "Continuous", 'Ranking']]
    await update.message.reply_text(
        "Привет! Я помогу тебе с A/B тестом. Скажи мне, какую метрику мы будем отслеживать.\n"
        "Отправь /cancel чтобы прекратить общение со мной.\n\n"
        "Discrete - данные, которые можно разбить на классы(пол, цвет глаз и т.д)\n\n"
        "Continuous - непрерывные данные(рост, вес и т.д)\n\n"
        "Ranking - ранговые данные(место в соревновании)",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Metric?"
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
        "Пожалуйста, пришлите мне ссылку на файл на гугл диске.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PATH


async def path_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected path and asks for a group column."""
    global PATH_M
    user = update.message.from_user
    PATH_M = update.message.text
    logger.info("Path of %s: %s", user.first_name, PATH_M)
    await update.message.reply_text(
        "Пожалуйста, напишите мне название колонки с группами.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return GROUP_COLUMN


async def group_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the group column and asks for value column."""
    global GROUP_COLUMN_M
    user = update.message.from_user
    GROUP_COLUMN_M = update.message.text
    logger.info("Group column of %s: %s", user.first_name, GROUP_COLUMN_M)
    await update.message.reply_text(
        "Пожалуйста, напишите мне название колонки со значениями.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return VALUES_COLUMN


async def value_column_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the info about value column."""
    global VALUES_COLUMN_M
    global DATAFRAME
    user = update.message.from_user
    VALUES_COLUMN_M = update.message.text
    logger.info("Value column of %s: %s", user.first_name, VALUES_COLUMN_M)

    url = PATH_M
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)
    if GROUP_COLUMN_M not in df.columns:
        await update.message.reply_text('Неверное название колонки с группами. Необходимо начать сначала.')
        return ConversationHandler.END
    if VALUES_COLUMN_M not in df.columns:
        await update.message.reply_text('Неверное название колонки со значениями. Необходимо начать сначала.')
        return ConversationHandler.END
    DATAFRAME, message, have_missing = library_for_ab.clear_data(df, GROUP_COLUMN_M, VALUES_COLUMN_M)
    if df.shape[0] == 0:
        await update.message.reply_text(message + ' Необходимо начать сначала.')
        return ConversationHandler.END
    if have_missing:
        reply_keyboard = [[ "Удалить", "Минимальное", 'Максимальное', 'Среднее', 'Медиану']]
        await update.message.reply_text(message + " \n\nЧто делать с пропущенными значениями? Выбери из предложенного или введи число, на которое заменим пропущенные значения:\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
        return DECISION_MISS
    p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
    try:
        await update.message.reply_text(message)
    except:
        pass
    final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
    if p_value < 0.05:
        await update.message.reply_text(message+'\n'
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    else:
        await update.message.reply_text(
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    return ConversationHandler.END

    
async def decision_missing_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global IS_OUTLIERS
    global DATAFRAME
    IS_OUTLIERS = False
    user = update.message.from_user
    decision_miss_value = update.message.text
    logger.info("decision_miss_value of %s: %s", user.first_name, decision_miss_value)
    DATAFRAME, message_miss = library_for_ab.missing_values(DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, decision_miss_value)
    dict_outliers = {}
    message = ''
    for group in DATAFRAME[GROUP_COLUMN_M].unique():
        message_outliers, flag_outliers = library_for_ab.get_outliers(DATAFRAME[DATAFRAME[GROUP_COLUMN_M] == group], VALUES_COLUMN_M)
        message += f'В группе {group}: ' + message_outliers + '\n'
        IS_OUTLIERS += flag_OUTLIERS
    if IS_OUTLIERS:
        reply_keyboard = [[ "Удалить", "Максимальное и минимальное", 'Среднее', 'Медиану', 'Оставить']]
        await update.message.reply_text(message_miss + "\n\n" + message + " \n\nЕсть выбросы. Подскажи, что мне с ними сделать? Выбери из предложенного:\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),)
        return CHANGE_OUTLIERS
    p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
    try:
        await update.message.reply_text(message)
    except:
        pass
    final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
    if p_value < 0.05:
        await update.message.reply_text(message+'\n'
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    else:
        await update.message.reply_text(
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    return ConversationHandler.END


async def change_outliers_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global IS_OUTLIERS
    global DATAFRAME
    user = update.message.from_user
    decision = update.message.text
    DATAFRAME, message, IS_OUTLIERS = library_for_ab.change_outliers(DATAFRAME[DATAFRAME[GROUP_COLUMN_M] == group], VALUES_COLUMN_M, decision)
    await update.message.reply_text(message)
    p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
    try:
        await update.message.reply_text(message)
    except:
        pass
    final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
    if p_value < 0.05:
        await update.message.reply_text(message+'\n'
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    else:
        await update.message.reply_text(
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    return ConversationHandler.END

async def p_value_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, DATAFRAME, GROUP_COLUMN_M, VALUES_COLUMN_M, IS_OUTLIERS)
    try:
        await update.message.reply_text(message)
    except:
        pass
    final_message = library_for_ab.get_conclusion(df, GROUP_COLUMN_M, VALUES_COLUMN_M, p_value)
    if p_value < 0.05:
        await update.message.reply_text(message+'\n'
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    else:
        await update.message.reply_text(
            'p_value: ' + str(p_value) + ' мощность: ' + str(power))
        await update.message.reply_text(final_message)
    return ConversationHandler.END

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

    application.add_handler(conv_handler_start)
    application.add_handler(conv_handler_sample_size)
    application.add_handler(conv_handler_ab_test)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
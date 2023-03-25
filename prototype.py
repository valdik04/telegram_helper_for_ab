import logging
import library_for_ab
from telegram import __version__ as TG_VER
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

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

METRIC, PATH, GROUP_COLUMN, VALUES_COLUMN = range(4)
METRIC_M, PATH_M, GROUP_COLUMN_M, VALUES_COLUMN_M = '', '', '', ''


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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


async def metric(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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


async def path(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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


async def group_column(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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


# async def skip_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Skips the location and asks for info about the user."""
#     user = update.message.from_user
#     logger.info("User %s did not send a location.", user.first_name)
#     await update.message.reply_text(
#         "You seem a bit paranoid! At last, tell me something about yourself."
#     )
#
#     return BIO
#

async def value_column(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the info about value column."""
    global VALUES_COLUMN_M
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
    df_clear = library_for_ab.clear_data(df, GROUP_COLUMN_M, VALUES_COLUMN_M)
    df = df_clear[0]
    message = df_clear[1]
    if df.shape[0] == 0:
        await update.message.reply_text(message + ' Необходимо начать сначала.')
        return ConversationHandler.END
    p_value, power, message, df = library_for_ab.get_p_value(METRIC_M, df, GROUP_COLUMN_M, VALUES_COLUMN_M)
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
    application = Application.builder().token("5871285096:AAFQqR0yppdxsW0tn29liLpwihrUHLFtXBw").build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            METRIC: [MessageHandler(filters.Regex("^(Discrete|Continuous|Ranking)$"), metric)],
            PATH: [MessageHandler(filters.Regex("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"), path)],
            GROUP_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, group_column)],
            VALUES_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, value_column)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
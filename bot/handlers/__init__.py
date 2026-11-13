from bot.handlers.students.base.start_cmd import register_start_cmd
from bot.handlers.students.base.help_cmd import register_help_cmd
from bot.handlers.students.base.menu_cmd import register_menu_cmd
from bot.handlers.students.base.logout_cmd import register_logout_cmd
from bot.handlers.students.base.profile_cmd import register_profile_cmd

from bot.handlers.students.account.auth_handler import register_authorize_handler_cmd
from bot.handlers.students.account.exit_handler import register_exit_handler_cmd

from bot.handlers.students.admin.admin_cmd import register_admin_cmd
from bot.handlers.students.admin.admin_callback import register_admin_callback_cmd

from bot.handlers.students.default.default_handler import register_default_handler_cmd

from bot.handlers.students.base.agreement import register_agreement_cmd
from bot.handlers.students.base.sql_query import register_sql_query_cmd

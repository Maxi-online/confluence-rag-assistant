from bot.handlers import (
		register_start_cmd,
		register_help_cmd,
		register_menu_cmd,
		register_logout_cmd,
		register_profile_cmd,

		register_admin_cmd,
		register_admin_callback_cmd,

		register_authorize_handler_cmd,
		register_exit_handler_cmd,
		register_default_handler_cmd,
		register_agreement_cmd,
		register_sql_query_cmd,
	)


def register_all_routers(dp):
	register_start_cmd(dp)
	register_help_cmd(dp)
	register_menu_cmd(dp)
	register_logout_cmd(dp)
	register_profile_cmd(dp)

	register_admin_cmd(dp)
	register_admin_callback_cmd(dp)

	register_authorize_handler_cmd(dp)
	register_exit_handler_cmd(dp)
	register_sql_query_cmd(dp)
	register_default_handler_cmd(dp)
	register_agreement_cmd(dp)

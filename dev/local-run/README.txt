local-run — 本机运行 aicheckword（与 git / 发版无关）

  start_api.bat              启动 FastAPI（8000，推荐，独立日志窗口）
  stop_api.bat               只停 API
  start_streamlit.bat        只启动 Streamlit 训练/管理 UI（8501）
  start_streamlit_lan.bat    Streamlit 局域网可访问
  start_all.bat              Web + API 一起启动
  stop_all.bat               停 Web + API
  restart_streamlit.bat      重启 Streamlit
  restart_all.bat            重启 Web + API
  preflight_check.bat        开工前 .cursor rules/skills 自检

测试环境：在 .env 设 AIWORD_ENV=test 后仍用 start_api.bat。

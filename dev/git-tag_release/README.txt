git-tag_release — 发版 tag 由 aiword 统一打

  请勿在本仓库单独打 v 版本 tag。
  请在同级 aiword 仓库运行:
    aiword\dev\git-tag_release\release.bat 1.0.5 "说明"

  release.bat 会同时 commit/push/tag aiword 与 aicheckword。

# Runbook: HighErrorRate

1. تحقق من الـlogs:
   docker compose logs zebra-observer --since 10m
2. تحقق من صحة الـDB وconnections.
3. افحص آخر تغيير للكود/موديل: git log --since="1 day" -- src/
4. إن لم تُحلّ المشكلة: rollback to last stable image:
   docker compose pull zebra-observer:stable && docker compose up -d
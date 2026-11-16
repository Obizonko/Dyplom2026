# data_small

Малі вхідні таблиці, які вручну перенесено з Drive, бо хмарна сесія не читає великі файли.

- `channel_flow_edges.csv`: 214 ребер пересилань між каналами (колонки `src_channel_id, src_channel, dst_channel_id, dst_channel, weight`).
  Джерело: `MyDrive/narratives-thesis/` (файл `channel_flow_edges.csv`, id 1fzslikgvo9zSSTSwwi6NGOfVvUE7Sn43), знімок графу 11.03.2026.
  Перенесення звірено: 214 рядків, 73 канали, id і назви збігаються з `results/02_eda/per_channel.csv`, відновлений файл за розміром у байтах (20 749) збігається з оригіналом.
  Напрямок: `src` канал-оригінал, `dst` канал, що переслав.

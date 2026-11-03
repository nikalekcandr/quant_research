# Перенос лаборатории и исполнителя на сервер

Зачем: ноутбук засыпает (30.10 и 03.11 потеряно 6 и 8 часов), интернет нестабилен, а 24/7 нагрузка ноутбуку вредна.
Сервер решает все три проблемы и заодно сокращает задержку до Kalshi с ~150–300 мс до единиц миллисекунд.

## 1. Какой сервер

| Требование | Почему | Минимум |
|---|---|---|
| Регион США, рядом с AWS us-east-2 (Огайо) | там стоят серверы Kalshi (за CloudFront) | Огайо > Чикаго ≈ Вирджиния/Нью-Йорк |
| 2 vCPU, 4 ГБ RAM | pandas + градиентный бустинг в зоопарке волатильности; 2 ГБ впритык | 2 vCPU / 4 ГБ |
| 40–80 ГБ диска | данные лаборатории растут на ~1–1,5 ГБ в месяц | 40 ГБ SSD |
| Ubuntu 22.04/24.04 | скрипты установки написаны под неё | — |

Варианты (цены меняются, проверь на сайте перед покупкой):

| Провайдер | Регион | Плюсы | Минусы |
|---|---|---|---|
| **AWS Lightsail** | us-east-2 (Огайо) | тот же регион AWS, что и Kalshi; фиксированная цена; простая панель | нужна зарубежная карта |
| Vultr / DigitalOcean | Чикаго, Нью-Джерси, Нью-Йорк | дёшево, почасовая оплата | 10–20 мс до Огайо (для наших сделок неважно) |
| Hetzner Cloud | Эшберн (Вирджиния) | обычно самый дешёвый | с июня 2026 цены выросли, а дешёвые тарифы часто недоступны |

**Практические вопросы:**
* **Оплата.** Карты российских банков у этих провайдеров не проходят: нужна зарубежная карта или провайдер с другими
  способами оплаты.
* **Юрисдикция.** Сервер в США решает задержку, но не меняет, кто ты как клиент Kalshi. Допуск к торговле определяется
  правилами Kalshi для твоей страны. Проверь их до реальных денег. Не используй сервер, чтобы обойти ограничения
  площадки: это прямой риск блокировки счёта вместе с деньгами.

## 2. Что будет работать на сервере

| Служба (systemd) | Что делает |
|---|---|
| `btcvol-lab` | лаборатория, только крылья W03–W06; перезапуск при любой остановке |
| `btcvol-shadow` | исполнитель в тени: настоящий стакан, без заявок |
| `btcvol-demo` | исполнитель на демо-бирже (включается после копирования ключа) |
| `btcvol-live.timer` | логгер курсового проекта каждые 10 минут |
| `btcvol-daily.timer` | ежедневное обновление данных в 06:20 UTC |

Задачи из `jobs/` (выгрузки истории, Windows-запуски) на сервере не выполняются: установка помечает их выполненными.
Реальная торговля на сервере сама не запускается никогда.

## 3. Пошагово

1. Создать сервер Ubuntu 24.04 (2 vCPU / 4 ГБ / 40+ ГБ) с SSH-ключом. Войти: `ssh ubuntu@<IP>`.
2. Установить всё одной командой:
   ```bash
   git clone https://github.com/nikalekcandr/quant_research.git && bash quant_research/deploy/server/setup.sh
   ```
   После мержа PR #7 в main; до этого — с веткой: `BRANCH=trader-o04-gamma bash quant_research/deploy/server/setup.sh`.
3. Перенести состояние лаборатории с компьютера. Задача `jobs/pack_for_server_now.py` уже собирает архив
   `_export\server_data.tar.gz`: журнал сделок, позиции, минутные данные, состояние исполнителя, **без ключей**.
   С Windows (PowerShell, из папки проекта):
   ```powershell
   scp _export\server_data.tar.gz ubuntu@<IP>:~/
   ```
   На сервере: `cd ~/quant_research && tar xzf ~/server_data.tar.gz && systemctl --user restart btcvol-lab`.
4. Ключ демо — только на сервер, по SSH:
   ```powershell
   scp <файл_ключа> ubuntu@<IP>:~/quant_research/data/trader/keys/demo/
   ssh ubuntu@<IP> "echo <id_ключа> > ~/quant_research/data/trader/keys/demo/key_id.txt && chmod 600 ~/quant_research/data/trader/keys/demo/* && systemctl --user enable --now btcvol-demo"
   ```
5. Остановить лабораторию на ноутбуке, чтобы не было двух копий (в PowerShell из папки проекта):
   `powershell -ExecutionPolicy Bypass -File uninstall_tasks.ps1` (снимает и логгер, и лабораторию).

## 4. Каждый день

```bash
bash ~/quant_research/deploy/server/status.sh      # службы, пульс лаборатории и исполнителей, тест Вальда
bash ~/quant_research/deploy/server/update.sh      # подтянуть код из GitHub и перезапустить
tail -f ~/quant_research/logs/btcvol-demo.out      # живой лог исполнителя
cd ~/quant_research && .venv/bin/python -m trader status --mode demo
```
Отчёт лаборатории `reports/lab.html` можно забрать: `scp ubuntu@<IP>:~/quant_research/reports/lab.html .`

## 5. Реальные деньги (1 контракт), когда будут выполнены условия из docs/WINGS.md §9

```bash
scp <prod_ключ> ubuntu@<IP>:~/quant_research/data/trader/keys/prod/       # + key_id.txt, chmod 600
ssh ubuntu@<IP>
cd ~/quant_research && touch data/trader/keys/ALLOW_LIVE
tmux new -s live '.venv/bin/python -m trader run --profile live_1contract --real-money'
```
Остановить: `.venv/bin/python -m trader kill` (снимет заявки и выйдет) или Ctrl+C в tmux.

## 6. Безопасность

* Вход только по SSH-ключу; пароль по SSH отключить (`PasswordAuthentication no`).
* Ключи Kalshi: `chmod 600`, только в `data/trader/keys/`, в git не попадают (`.gitignore`).
* На ключе для реальных денег — только права торговли, без вывода средств (если площадка позволяет так ограничить).
* Резервная копия журнала: `data/lab/ledger.csv` и `data/trader/journal_*.jsonl` раз в день скачивать на ноутбук
  или пушить в отдельный приватный репозиторий.

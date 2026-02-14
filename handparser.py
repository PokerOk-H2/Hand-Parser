#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handparser.py — парсер историй раздач PokerOK и анализатор статистики по герою.

Ограничения:
- Формат HH взят как обобщённый: похож на PokerStars / GG / PokerOK.
- Регулярки и парсер нужно будет немного подогнать под реальный текст PokerOK.
- Всё в одном файле, только стандартная библиотека.
"""

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Модели данных
# ----------------------------

@dataclass
class Action:
    player: str
    street: str  # 'preflop', 'flop', 'turn', 'river'
    action: str  # 'fold', 'call', 'raise', 'bet', 'check', 'allin', 'posts_blind', ...
    amount: Optional[float] = None


@dataclass
class Player:
    name: str
    seat: Optional[int] = None
    stack: Optional[float] = None
    position_preflop: Optional[str] = None  # 'SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'


@dataclass
class Hand:
    hand_id: str
    datetime: Optional[datetime]
    game_type: Optional[str]
    stakes_raw: Optional[str]
    stakes_label: Optional[str]
    table_name: Optional[str]
    max_players: Optional[int]
    players: List[Player]
    hero_name: Optional[str]
    hero_hole_cards: Optional[List[str]]
    board_flop: Optional[List[str]]
    board_turn: Optional[List[str]]
    board_river: Optional[List[str]]
    actions: List[Action]
    hero_net_win: float
    hero_went_to_showdown: bool
    hero_won_showdown: Optional[bool]
    total_pot: Optional[float] = None


@dataclass
class StatsSummary:
    hands_total: int = 0
    hands_with_hero: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    threebet_hands: int = 0
    threebet_spots: int = 0
    bets_raises_postflop: int = 0
    calls_postflop: int = 0
    cbet_flop_success: int = 0
    cbet_flop_spots: int = 0
    wtsd_hands: int = 0
    wsd_wins: int = 0
    total_profit: float = 0.0
    bb_profit_sum: float = 0.0  # суммарный профит в бигблайндах
    bb_hands_count: int = 0     # количество рук, где известен BB

    @property
    def vpip(self) -> float:
        return (self.vpip_hands / self.hands_with_hero * 100) if self.hands_with_hero else 0.0

    @property
    def pfr(self) -> float:
        return (self.pfr_hands / self.hands_with_hero * 100) if self.hands_with_hero else 0.0

    @property
    def threebet(self) -> float:
        return (self.threebet_hands / self.threebet_spots * 100) if self.threebet_spots else 0.0

    @property
    def af(self) -> float:
        if self.calls_postflop == 0:
            return float(self.bets_raises_postflop)
        return self.bets_raises_postflop / self.calls_postflop

    @property
    def cbet_flop(self) -> float:
        return (self.cbet_flop_success / self.cbet_flop_spots * 100) if self.cbet_flop_spots else 0.0

    @property
    def wtsd(self) -> float:
        return (self.wtsd_hands / self.hands_with_hero * 100) if self.hands_with_hero else 0.0

    @property
    def wsd(self) -> float:
        return (self.wsd_wins / self.wtsd_hands * 100) if self.wtsd_hands else 0.0

    @property
    def bb_per_100(self) -> float:
        if not self.bb_hands_count:
            return 0.0
        # profit_bb / (hands/100) = profit_bb * 100 / hands
        return self.bb_profit_sum * 100.0 / self.bb_hands_count


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PokerOK hand history parser and hero stats calculator."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to hand history file or directory."
    )
    parser.add_argument(
        "-H", "--hero", required=True,
        help="Hero name (nickname) to analyze."
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively search for hand history files in directory."
    )
    parser.add_argument(
        "-s", "--stakes", default=None,
        help='Comma-separated list of stakes labels to include, e.g. "NL10,NL25".'
    )
    parser.add_argument(
        "--min-hands", type=int, default=30,
        help="Minimum number of hands to show stakes in summary (per stakes)."
    )
    parser.add_argument(
        "-o", "--out", default=None,
        help="Output file for stats (CSV or JSON depending on --out-format)."
    )
    parser.add_argument(
        "--out-format", choices=["summary", "csv", "json"], default="summary",
        help="Output format: human-readable summary (default), csv, or json."
    )
    parser.add_argument(
        "--encoding", default=None,
        help="Force file encoding (default: try utf-8 then cp1251)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run built-in self test and exit."
    )
    return parser.parse_args()


# ----------------------------
# Работа с файлами
# ----------------------------

def collect_files(input_path: str, recursive: bool) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files: List[Path] = []
    if path.is_file():
        files.append(path)
    else:
        if recursive:
            for p in path.rglob("*"):
                if p.is_file():
                    files.append(p)
        else:
            for p in path.iterdir():
                if p.is_file():
                    files.append(p)

    # можно фильтровать по расширению, но оставим всё текстовое
    return files


def read_file(path: Path, encoding: Optional[str] = None) -> str:
    encodings_to_try = [encoding] if encoding else ["utf-8", "cp1251"]
    last_exc: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            with path.open("r", encoding=enc) as f:
                return f.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logging.debug("Failed to read %s with encoding %s: %s", path, enc, exc)
    raise last_exc if last_exc else IOError(f"Failed to read file: {path}")


# ----------------------------
# Парсер раздач
# ----------------------------

def split_into_hands(raw_text: str) -> List[str]:
    """
    Делит общий текст на отдельные раздачи.
    Предполагаем, что каждая раздача начинается с строки, содержащей 'Hand #'.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    pattern = re.compile(r'(?=^.*Hand\s+#\d+)', re.MULTILINE)
    parts = pattern.split(raw_text)

    hands = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # проверяем, что там действительно есть "Hand #"
        if "Hand #" not in part:
            # возможно, это заголовок до первой раздачи — пропускаем
            continue
        hands.append(part)
    return hands


HEADER_REGEX = re.compile(
    r"Hand\s+#(?P<hand_id>\d+).*?(?P<game_type>Hold'em|Omaha|PLO|NLH|NL Hold'em|PL Omaha)?.*?"
    r"\((?P<stakes>[^)]+)\).*?- (?P<datetime>.+)$"
)

TABLE_REGEX = re.compile(
    r"^Table\s+'?(?P<table_name>[^']+)'?\s+(?P<max_players>\d+)-max", re.IGNORECASE
)

SEAT_REGEX = re.compile(
    r"^Seat\s+(?P<seat>\d+):\s+(?P<name>.+?)\s+\((?P<stack>[\$\d\.\,]+)\s+in chips\)",
    re.IGNORECASE
)

DEALT_REGEX = re.compile(
    r"^Dealt to\s+(?P<name>.+?)\s+\[(?P<cards>[^\]]+)\]", re.IGNORECASE
)

BOARD_FLOP_REGEX = re.compile(
    r"^\*\*\*\s+FLOP\s+\*\*\*\s+\[(?P<cards>[^\]]+)\]", re.IGNORECASE
)

BOARD_TURN_REGEX = re.compile(
    r"^\*\*\*\s+TURN\s+\*\*\*\s+\[(?P<flop>[^\]]+)\]\s+\[(?P<card>[^\]]+)\]", re.IGNORECASE
)

BOARD_RIVER_REGEX = re.compile(
    r"^\*\*\*\s+RIVER\s+\*\*\*\s+\[(?P<flop_turn>[^\]]+)\]\s+\[(?P<card>[^\]]+)\]", re.IGNORECASE
)

TOTAL_POT_REGEX = re.compile(
    r"^Total pot\s+([\$\d\.\,]+)", re.IGNORECASE
)

COLLECTED_REGEX = re.compile(
    r"^(?P<name>.+?)\s+collected\s+([\$\d\.\,]+)\s+from pot", re.IGNORECASE
)

WON_REGEX = re.compile(
    r"^(?P<name>.+?)\s+won\s+([\$\d\.\,]+)", re.IGNORECASE
)


def _parse_money(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    m = re.search(r"([\d\.]+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def stakes_to_label(stakes: Optional[str]) -> Optional[str]:
    """
    Преобразует строку типа '$0.05/$0.10' в 'NL10' (примерно).
    Если распарсить не удаётся — возвращает None.
    """
    if not stakes:
        return None
    # ищем второе число — размер BB
    m = re.search(r"[/ ]([\$€]?[\d\.]+)", stakes)
    if not m:
        # fallback: любое число
        m = re.search(r"([\$€]?[\d\.]+)$", stakes)
    if not m:
        return None
    bb = _parse_money(m.group(1))
    if bb is None:
        return None
    label = int(round(bb * 100))
    return f"NL{label}"


def parse_header_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[datetime]]:
    """
    Возвращает (hand_id, game_type, stakes, dt)
    """
    m = HEADER_REGEX.search(line)
    if not m:
        return None, None, None, None

    hand_id = m.group("hand_id")
    game_type = m.group("game_type")
    stakes = m.group("stakes")
    dt_raw = m.group("datetime").strip()

    dt = None
    # пробуем несколько форматов, реальные форматы HH могут отличаться
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_raw, fmt)
            break
        except ValueError:
            continue

    return hand_id, game_type, stakes, dt


def parse_hand(text: str, hero_name: str) -> Optional[Hand]:
    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]

    if not lines:
        return None

    # ----- header -----
    hand_id = None
    game_type = None
    stakes_raw = None
    dt = None
    table_name = None
    max_players = None
    players: List[Player] = []
    hero_hole_cards: Optional[List[str]] = None
    board_flop = None
    board_turn = None
    board_river = None
    actions: List[Action] = []
    hero_net_win = 0.0
    hero_went_to_showdown = False
    hero_won_showdown = None
    total_pot = None

    # 1. Header
    hand_id, game_type, stakes_raw, dt = parse_header_line(lines[0])

    # 2. Table / seats / players
    street = "preflop"
    hero_name_lower = hero_name.lower()
    hero_sat_in = False

    for line in lines[1:]:
        # конец заголовков, начало действий может быть по-разному
        if line.startswith("*** HOLE CARDS ***"):
            street = "preflop"
            continue
        if line.startswith("*** FLOP ***"):
            street = "flop"
            m = BOARD_FLOP_REGEX.match(line)
            if m:
                board_flop = [c.strip() for c in m.group("cards").split()]
            continue
        if line.startswith("*** TURN ***"):
            street = "turn"
            m = BOARD_TURN_REGEX.match(line)
            if m:
                # m.group("flop") содержит flop, но мы уже могу иметь board_flop
                card = m.group("card").strip()
                if board_flop:
                    board_turn = board_flop + [card]
                else:
                    board_turn = [card]
            continue
        if line.startswith("*** RIVER ***"):
            street = "river"
            m = BOARD_RIVER_REGEX.match(line)
            if m:
                card = m.group("card").strip()
                prev = board_turn or board_flop or []
                board_river = prev + [card]
            continue
        if line.startswith("*** SHOW DOWN ***"):
            hero_went_to_showdown = True
            continue
        if line.startswith("*** SUMMARY ***"):
            # дальнейшие строки — итоги, но часть мы уже собираем
            continue

        # Table line
        m_table = TABLE_REGEX.match(line)
        if m_table:
            table_name = m_table.group("table_name").strip()
            try:
                max_players = int(m_table.group("max_players"))
            except (TypeError, ValueError):
                max_players = None
            continue

        # Seat / players
        m_seat = SEAT_REGEX.match(line)
        if m_seat:
            seat = int(m_seat.group("seat"))
            name = m_seat.group("name").strip()
            stack = _parse_money(m_seat.group("stack"))
            players.append(Player(name=name, seat=seat, stack=stack))
            if name.lower() == hero_name_lower:
                hero_sat_in = True
            continue

        # Dealt to hero
        m_dealt = DEALT_REGEX.match(line)
        if m_dealt:
            name = m_dealt.group("name").strip()
            if name.lower() == hero_name_lower:
                cards_str = m_dealt.group("cards")
                hero_hole_cards = cards_str.split()
            continue

        # Total pot
        m_total = TOTAL_POT_REGEX.match(line)
        if m_total:
            total_pot = _parse_money(m_total.group(1))
            continue

        # Collected / won (results)
        m_collect = COLLECTED_REGEX.match(line)
        if m_collect:
            name = m_collect.group("name").strip()
            amount = _parse_money(m_collect.group(0))
            if amount is not None and name.lower() == hero_name_lower:
                hero_net_win += amount
                # если это в шоудауне — проставим позже по флагу
                hero_won_showdown = hero_went_to_showdown or hero_won_showdown
            continue

        m_won = WON_REGEX.match(line)
        if m_won:
            name = m_won.group("name").strip()
            amount = _parse_money(m_won.group(0))
            if amount is not None and name.lower() == hero_name_lower:
                hero_net_win += amount
                hero_won_showdown = hero_went_to_showdown or hero_won_showdown
            continue

        # Actions
        act = parse_action_line(line, street)
        if act:
            actions.append(act)
            # учёт вложенных денег героем:
            if act.player.lower() == hero_name_lower:
                if act.action in {"call", "bet", "raise", "allin", "posts_blind"} and act.amount:
                    hero_net_win -= act.amount
            continue

    if not hero_sat_in:
        # Героя нет в раздаче — можно не учитывать
        return None

    # Позиции префлоп задаём по порядку сидений
    assign_preflop_positions(players)

    stakes_label = stakes_to_label(stakes_raw)

    return Hand(
        hand_id=hand_id or "",
        datetime=dt,
        game_type=game_type,
        stakes_raw=stakes_raw,
        stakes_label=stakes_label,
        table_name=table_name,
        max_players=max_players,
        players=players,
        hero_name=hero_name,
        hero_hole_cards=hero_hole_cards,
        board_flop=board_flop,
        board_turn=board_turn,
        board_river=board_river,
        actions=actions,
        hero_net_win=hero_net_win,
        hero_went_to_showdown=hero_went_to_showdown,
        hero_won_showdown=hero_won_showdown,
        total_pot=total_pot,
    )


def assign_preflop_positions(players: List[Player]) -> None:
    """
    Очень упрощённое присвоение позиций по seat: SB, BB, UTG, ... BTN.
    Предполагаем, что seat — это посадка за столом по часовой.
    """
    if not players:
        return
    # сортируем по seat
    players_sorted = sorted(players, key=lambda p: (p.seat if p.seat is not None else 999))
    n = len(players_sorted)
    if n == 0:
        return

    # Индексы: 0-SB, 1-BB, далее UTG, MP, CO, BTN (упрощённо)
    positions_ring = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
    for i, player in enumerate(players_sorted):
        idx = i if i < len(positions_ring) else len(positions_ring) - 1
        player.position_preflop = positions_ring[idx]


def parse_action_line(line: str, street: str) -> Optional[Action]:
    """
    Простейший парсер строк действия:
    Примеры:
    - 'Player1: folds'
    - 'Player2: checks'
    - 'Player3: calls $0.20'
    - 'Player4: bets $0.30'
    - 'Player5: raises to $0.50'
    - 'Player6: posts small blind $0.05'
    - 'Player7: posts big blind $0.10'
    """
    # быстро отсекаем явно не action
    if ":" not in line:
        return None
    if line.startswith("Seat "):
        return None

    m = re.match(r"^(?P<player>[^:]+):\s+(?P<rest>.+)$", line)
    if not m:
        return None

    player = m.group("player").strip()
    rest = m.group("rest").strip().lower()

    action_type = None
    amount = None

    if "posts" in rest and "blind" in rest:
        action_type = "posts_blind"
        m_amt = re.search(r"([\$€]?\d+(\.\d+)?)", rest)
        if m_amt:
            amount = _parse_money(m_amt.group(1))
    elif rest.startswith("folds"):
        action_type = "fold"
    elif rest.startswith("checks"):
        action_type = "check"
    elif rest.startswith("calls"):
        action_type = "call"
        m_amt = re.search(r"([\$€]?\d+(\.\d+)?)", rest)
        if m_amt:
            amount = _parse_money(m_amt.group(1))
    elif rest.startswith("bets"):
        action_type = "bet"
        m_amt = re.search(r"([\$€]?\d+(\.\d+)?)", rest)
        if m_amt:
            amount = _parse_money(m_amt.group(1))
    elif rest.startswith("raises"):
        action_type = "raise"
        m_amt = re.search(r"to\s+([\$€]?\d+(\.\d+)?)", rest)
        if m_amt:
            amount = _parse_money(m_amt.group(1))
    elif "all-in" in rest or "all in" in rest:
        # упростим: считаем как allin (агрессивное действие)
        action_type = "allin"
        m_amt = re.search(r"([\$€]?\d+(\.\d+)?)", rest)
        if m_amt:
            amount = _parse_money(m_amt.group(1))

    if not action_type:
        return None

    return Action(player=player, street=street, action=action_type, amount=amount)


# ----------------------------
# Анализ одной раздачи (для статов)
# ----------------------------

def analyze_hand_for_stats(hand: Hand, hero_name: str) -> Dict[str, any]:
    hero_name_lower = hero_name.lower()

    # preflop
    preflop_actions = [a for a in hand.actions if a.street == "preflop"]
    flop_actions = [a for a in hand.actions if a.street == "flop"]
    turn_actions = [a for a in hand.actions if a.street == "turn"]
    river_actions = [a for a in hand.actions if a.street == "river"]

    hero_preflop_actions = [a for a in preflop_actions if a.player.lower() == hero_name_lower]
    hero_postflop_actions = [a for a in hand.actions if a.street in ("flop", "turn", "river")
                             and a.player.lower() == hero_name_lower]

    # VPIP
    vpip = any(a.action in {"call", "bet", "raise", "allin"} for a in hero_preflop_actions)

    # PFR
    pfr = any(a.action in {"bet", "raise", "allin"} for a in hero_preflop_actions)

    # 3-bet (и споты)
    threebet = False
    threebet_spot = False
    raise_count = 0
    hero_first_action_index = None

    for idx, a in enumerate(preflop_actions):
        if a.player.lower() == hero_name_lower and hero_first_action_index is None:
            hero_first_action_index = idx
        if a.action in {"bet", "raise", "allin"}:
            # агрессивное действие
            if a.player.lower() == hero_name_lower:
                if raise_count >= 1:
                    threebet = True
            raise_count += 1

    if hero_first_action_index is not None:
        # был ли raise до первой реакции героя?
        for a in preflop_actions[:hero_first_action_index]:
            if a.action in {"bet", "raise", "allin"} and a.player.lower() != hero_name_lower:
                threebet_spot = True
                break

    # AF (postflop)
    bets_raises_postflop = 0
    calls_postflop = 0
    for a in hero_postflop_actions:
        if a.action in {"bet", "raise", "allin"}:
            bets_raises_postflop += 1
        elif a.action == "call":
            calls_postflop += 1

    # C-bet flop
    # герой - последний агрессор на префлопе?
    last_agg_player = None
    for a in preflop_actions:
        if a.action in {"bet", "raise", "allin"}:
            last_agg_player = a.player.lower()

    hero_preflop_agg = (last_agg_player == hero_name_lower)
    saw_flop = bool(flop_actions) and not any(
        a.player.lower() == hero_name_lower and a.action == "fold" for a in preflop_actions
    )
    cbet_flop_spot = hero_preflop_agg and saw_flop

    hero_cbet_flop = False
    if cbet_flop_spot:
        # герой делал bet/raise/allin на флопе?
        for a in flop_actions:
            if a.player.lower() == hero_name_lower and a.action in {"bet", "raise", "allin"}:
                hero_cbet_flop = True
                break

    # WTSD / W$SD берём из Hand
    wtsd = hand.hero_went_to_showdown
    wsd_win = bool(hand.hero_won_showdown)

    # BB в этой раздаче
    bb = None
    if hand.stakes_raw:
        bb = extract_bb_from_stakes(hand.stakes_raw)

    return {
        "vpip": vpip,
        "pfr": pfr,
        "threebet": threebet,
        "threebet_spot": threebet_spot,
        "bets_raises_postflop": bets_raises_postflop,
        "calls_postflop": calls_postflop,
        "cbet_flop_spot": cbet_flop_spot,
        "cbet_flop": hero_cbet_flop,
        "wtsd": wtsd,
        "wsd_win": wsd_win,
        "bb": bb,
    }


def extract_bb_from_stakes(stakes: str) -> Optional[float]:
    """
    Из строки типа '$0.05/$0.10' возвращаем 0.10.
    """
    # ищем второе число
    m = re.search(r"/\s*([\$€]?\d+(\.\d+)?)", stakes)
    if not m:
        return None
    return _parse_money(m.group(1))


# ----------------------------
# Подсчёт статистики
# ----------------------------

def compute_stats(hands: List[Hand], hero_name: str) -> Tuple[StatsSummary, Dict[str, StatsSummary]]:
    """
    Возвращает глобальные статы и статы по лимитам (stakes_label).
    """
    global_stats = StatsSummary()
    by_stakes: Dict[str, StatsSummary] = {}

    hero_name_lower = hero_name.lower()

    for hand in hands:
        if not hand.hero_name or hand.hero_name.lower() != hero_name_lower:
            # на всякий случай
            continue

        global_stats.hands_total += 1
        global_stats.hands_with_hero += 1

        stakes_label = hand.stakes_label or "UNKNOWN"
        if stakes_label not in by_stakes:
            by_stakes[stakes_label] = StatsSummary()
        st = by_stakes[stakes_label]

        st.hands_total += 1
        st.hands_with_hero += 1

        # анализируем раздачу
        analysis = analyze_hand_for_stats(hand, hero_name)

        if analysis["vpip"]:
            global_stats.vpip_hands += 1
            st.vpip_hands += 1
        if analysis["pfr"]:
            global_stats.pfr_hands += 1
            st.pfr_hands += 1
        if analysis["threebet_spot"]:
            global_stats.threebet_spots += 1
            st.threebet_spots += 1
            if analysis["threebet"]:
                global_stats.threebet_hands += 1
                st.threebet_hands += 1

        global_stats.bets_raises_postflop += analysis["bets_raises_postflop"]
        global_stats.calls_postflop += analysis["calls_postflop"]
        st.bets_raises_postflop += analysis["bets_raises_postflop"]
        st.calls_postflop += analysis["calls_postflop"]

        if analysis["cbet_flop_spot"]:
            global_stats.cbet_flop_spots += 1
            st.cbet_flop_spots += 1
            if analysis["cbet_flop"]:
                global_stats.cbet_flop_success += 1
                st.cbet_flop_success += 1

        if analysis["wtsd"]:
            global_stats.wtsd_hands += 1
            st.wtsd_hands += 1
            if analysis["wsd_win"]:
                global_stats.wsd_wins += 1
                st.wsd_wins += 1

        # профит
        global_stats.total_profit += hand.hero_net_win
        st.total_profit += hand.hero_net_win

        bb = analysis["bb"]
        if bb is not None and bb > 0:
            profit_bb = hand.hero_net_win / bb
            global_stats.bb_profit_sum += profit_bb
            global_stats.bb_hands_count += 1
            st.bb_profit_sum += profit_bb
            st.bb_hands_count += 1

    return global_stats, by_stakes


# ----------------------------
# Вывод и экспорт
# ----------------------------

def print_summary(global_stats: StatsSummary, by_stakes: Dict[str, StatsSummary], min_hands: int) -> None:
    print("=== GLOBAL STATS ===")
    print(f"Hands with hero:     {global_stats.hands_with_hero}")
    print(f"Total profit:        {global_stats.total_profit:.2f}")
    print(f"BB/100:              {global_stats.bb_per_100:.2f}")
    print(f"VPIP:                {global_stats.vpip:.2f}%")
    print(f"PFR:                 {global_stats.pfr:.2f}%")
    print(f"3-bet:               {global_stats.threebet:.2f}% "
          f"(spots={global_stats.threebet_spots})")
    print(f"Aggression factor:   {global_stats.af:.2f}")
    print(f"C-bet flop:          {global_stats.cbet_flop:.2f}% "
          f"(spots={global_stats.cbet_flop_spots})")
    print(f"WTSD:                {global_stats.wtsd:.2f}%")
    print(f"W$SD:                {global_stats.wsd:.2f}%")
    print()

    if by_stakes:
        print("=== STATS BY STAKES ===")
        header = (
            f"{'Stakes':<8} {'Hands':>7} {'VPIP%':>7} {'PFR%':>7} "
            f"{'3bet%':>7} {'BB/100':>8} {'Profit':>10}"
        )
        print(header)
        print("-" * len(header))
        for stakes_label, st in sorted(by_stakes.items(), key=lambda kv: kv[0]):
            if st.hands_with_hero < min_hands:
                continue
            print(
                f"{stakes_label:<8} "
                f"{st.hands_with_hero:>7} "
                f"{st.vpip:>7.2f} "
                f"{st.pfr:>7.2f} "
                f"{st.threebet:>7.2f} "
                f"{st.bb_per_100:>8.2f} "
                f"{st.total_profit:>10.2f}"
            )


def export_csv(path: str, by_stakes: Dict[str, StatsSummary]) -> None:
    fieldnames = [
        "stakes",
        "hands_with_hero",
        "vpip",
        "pfr",
        "threebet",
        "threebet_spots",
        "bb_per_100",
        "total_profit",
        "cbet_flop",
        "cbet_flop_spots",
        "wtsd",
        "wsd",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stakes_label, st in sorted(by_stakes.items(), key=lambda kv: kv[0]):
            if not st.hands_with_hero:
                continue
            writer.writerow(
                {
                    "stakes": stakes_label,
                    "hands_with_hero": st.hands_with_hero,
                    "vpip": f"{st.vpip:.2f}",
                    "pfr": f"{st.pfr:.2f}",
                    "threebet": f"{st.threebet:.2f}",
                    "threebet_spots": st.threebet_spots,
                    "bb_per_100": f"{st.bb_per_100:.2f}",
                    "total_profit": f"{st.total_profit:.2f}",
                    "cbet_flop": f"{st.cbet_flop:.2f}",
                    "cbet_flop_spots": st.cbet_flop_spots,
                    "wtsd": f"{st.wtsd:.2f}",
                    "wsd": f"{st.wsd:.2f}",
                }
            )


def export_json(path: str, global_stats: StatsSummary, by_stakes: Dict[str, StatsSummary]) -> None:
    def stats_to_dict(st: StatsSummary) -> Dict[str, any]:
        return {
            "hands_total": st.hands_total,
            "hands_with_hero": st.hands_with_hero,
            "vpip": st.vpip,
            "pfr": st.pfr,
            "threebet": st.threebet,
            "threebet_spots": st.threebet_spots,
            "af": st.af,
            "cbet_flop": st.cbet_flop,
            "cbet_flop_spots": st.cbet_flop_spots,
            "wtsd": st.wtsd,
            "wsd": st.wsd,
            "total_profit": st.total_profit,
            "bb_per_100": st.bb_per_100,
        }

    data = {
        "global": stats_to_dict(global_stats),
        "by_stakes": {stakes: stats_to_dict(st) for stakes, st in by_stakes.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------
# Self-test
# ----------------------------

SELF_TEST_TEXT = """
PokerOK - Hand #123456789: Hold'em No Limit ($0.05/$0.10) - 2025/01/01 20:30:00
Table 'Alpha' 6-max
Seat 1: Villain1 ($10 in chips)
Seat 2: HeroNick ($10 in chips)
Seat 3: Villain2 ($10 in chips)

*** HOLE CARDS ***
HeroNick: posts small blind $0.05
Villain1: posts big blind $0.10
Dealt to HeroNick [As Kd]
HeroNick: raises to $0.30
Villain2: folds
Villain1: calls $0.20

*** FLOP *** [2c 3d 9h]
Villain1: checks
HeroNick: bets $0.40
Villain1: calls $0.40

*** TURN *** [2c 3d 9h] [Qh]
Villain1: checks
HeroNick: bets $1.00
Villain1: folds

*** SUMMARY ***
Total pot $1.60 | Rake $0.05
HeroNick collected $1.55 from pot

PokerOK - Hand #123456790: Hold'em No Limit ($0.05/$0.10) - 2025/01/01 20:32:00
Table 'Alpha' 6-max
Seat 1: Villain1 ($10 in chips)
Seat 2: HeroNick ($11.55 in chips)
Seat 3: Villain2 ($10 in chips)

*** HOLE CARDS ***
HeroNick: posts small blind $0.05
Villain1: posts big blind $0.10
Dealt to HeroNick [7c 2d]
HeroNick: folds

*** SUMMARY ***
Total pot $0.15 | Rake $0.00
Villain1 collected $0.15 from pot
"""


def run_self_test() -> None:
    hero = "HeroNick"
    hands_text = split_into_hands(SELF_TEST_TEXT)
    print(f"Self-test: found {len(hands_text)} hands")
    hands: List[Hand] = []
    for htxt in hands_text:
        hand = parse_hand(htxt, hero)
        if hand:
            hands.append(hand)
    print(f"Parsed hands with hero: {len(hands)}")

    global_stats, by_stakes = compute_stats(hands, hero)
    print_summary(global_stats, by_stakes, min_hands=0)


# ----------------------------
# main
# ----------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(message)s",
    )

    if args.self_test:
        run_self_test()
        return

    hero_name = args.hero
    files = collect_files(args.input, args.recursive)

    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    logging.info("Found %d file(s).", len(files))

    all_hands: List[Hand] = []
    for path in files:
        try:
            text = read_file(path, encoding=args.encoding)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to read %s: %s", path, exc)
            continue
        hands_text = split_into_hands(text)
        logging.info("File %s: found %d hands", path, len(hands_text))
        for htxt in hands_text:
            try:
                hand = parse_hand(htxt, hero_name)
            except Exception as exc:  # noqa: BLE001
                if args.debug:
                    logging.exception("Failed to parse hand in %s: %s", path, exc)
                else:
                    logging.warning("Failed to parse hand in %s: %s", path, exc)
                continue
            if hand:
                all_hands.append(hand)

    if not all_hands:
        print("No hands with hero found.", file=sys.stderr)
        sys.exit(1)

    logging.info("Total parsed hands with hero: %d", len(all_hands))

    # фильтр по лимитам, если задан
    if args.stakes:
        wanted = {s.strip().upper() for s in args.stakes.split(",") if s.strip()}
        all_hands = [h for h in all_hands if (h.stakes_label or "").upper() in wanted]
        if not all_hands:
            print("No hands after stakes filter.", file=sys.stderr)
            sys.exit(1)

    global_stats, by_stakes = compute_stats(all_hands, hero_name)

    if args.out_format == "summary":
        print_summary(global_stats, by_stakes, min_hands=args.min_hands)
    elif args.out_format == "csv":
        if not args.out:
            print("Please specify --out for csv output.", file=sys.stderr)
            sys.exit(1)
        export_csv(args.out, by_stakes)
        print(f"CSV exported to {args.out}")
    elif args.out_format == "json":
        if not args.out:
            print("Please specify --out for json output.", file=sys.stderr)
            sys.exit(1)
        export_json(args.out, global_stats, by_stakes)
        print(f"JSON exported to {args.out}")


if __name__ == "__main__":
    main()
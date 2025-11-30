from ortools.sat.python import cp_model
import time
import json
import sys
from collections import defaultdict
import streamlit as st

st.set_page_config(page_title="Pokemon Sleep Solver", layout="wide")

class PokemonDeploymentSolver:
    def __init__(self, user_inputs):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # 定数の定義
        self.days = user_inputs["days"]
        self.num_hours = 24 * self.days  # 7日間 (24h * 7)
        self.day_switch_time = 9
        self.bedin_time = 24
        self.today = user_inputs["today"]  # 0: 月曜日, 1: 火曜日, 2: 水曜日, 3: 木曜日, 4: 金曜日, 5: 土曜日, 6: 日曜日
        self.start_time = 9
        self.start_time += self.today * 24
        self.max_active_pokemon = 5  # 最大同時動員数
        self.dish_week = user_inputs["dish"]
        self.berry_liked = user_inputs["berries"]
        base_cook_time = [9, 15, 20]
        self.evolution_num = 0
        self.designated_pokemon = []
        self.forbidden_pokemon = []
        self.pot_capacity = 69  # 料理の最大容量
        self.stamina_threshold = 30
        self.use_additional_ingredient = False  # 追加食材を使用するかどうか
        self.consider_dish_level = True
        self.use_seed_pokemon = False
        self.dish_energy_multiplier = 1.0
        self.skill_energy_multiplier = 1.0
        # 初期在庫
        self.initial_stock = user_inputs["stock"]
        self.final_stock = {
            "キノコ": 0,
            "卵": 0,
            "イモ": 0,
            "リンゴ": 0,
            "ハーブ": 0,
            "肉": 0,
            "牛乳": 0,
            "ミツ": 0,
            "オイル": 0,
            "ジンジャー": 0,
            "トマト": 0,
            "カカオ": 0,
            "大豆": 0,
            "コーヒー": 0,
        }
        self.storage_limit = 999  # 食材の最大所持量
        # ポケモンリストと収集能力 (1時間あたりに修正)
        with open("pokemon_data.json", encoding="utf-8_sig") as f:
            self.pokemon_data = json.load(f)

        with open("seed_pokemon_data.json", encoding="utf-8_sig") as f:
            self.seed_pokemon = json.load(f)

        with open("future_pokemon_data.json", encoding="utf-8_sig") as f:
            self.future_pokemon = json.load(f)

        self.pokemon_data = {**self.pokemon_data, **self.seed_pokemon, **self.future_pokemon}

        for pokemon in self.pokemon_data.values():
            pokemon["berries"] *= 36
            for ing_name in pokemon["ingredients"].keys():
                pokemon["ingredients"][ing_name] *= 100
                pokemon["ingredients"][ing_name] = int(pokemon["ingredients"][ing_name] / 24)

        self.berries = {
            "fire": 27,
            "water": 31,
            "grass": 30,
            "poison": 32,
            "earth": 29,
            "electric": 25,
            "ice": 32,
            "dragon": 35,
            "ghost": 26,
            "fairy": 26,
            "bird": 24,
            "normal": 28,
            "bug": 24,
            "rock": 30,
            "esper": 26,
            "fight": 27,
            "dark": 31,
            "steel": 33,
        }
        for p in self.pokemon_data:
            if self.pokemon_data[p]["type"] in self.berry_liked:
                self.pokemon_data[p]["berries"] = self.pokemon_data[p]["berries"] * 2
            self.pokemon_data[p]["berries"] = int(
                self.pokemon_data[p]["berries"]
                / self.berries[self.pokemon_data[p]["type"]]
                * max(
                    self.berries[self.pokemon_data[p]["type"]]
                    + self.pokemon_data[p]["level"]
                    - 1,
                    self.berries[self.pokemon_data[p]["type"]]
                    * 1.025 ** (self.pokemon_data[p]["level"] - 1),
                )
            )
        self.cook_time = [
            b + 24 * n
            for n in range(self.num_hours // 24)
            for b in base_cook_time
            if b + 24 * n >= self.start_time
        ]

        self.ingredients = {
            "リンゴ": 90,
            "牛乳": 98,
            "肉": 103,
            "ミツ": 101,
            "オイル": 121,
            "トマト": 110,
            "ハーブ": 130,
            "大豆": 100,
            "卵": 115,
            "カカオ": 151,
            "イモ": 124,
            "コーヒー": 153,
            "ジンジャー": 109,
            "キノコ": 167,
            "コーン": 140,
            "ネギ": 185,
            "シッポ": 342,
            "カボチャ": 250,
            "アボカド": 162,
        }

        for k, v in self.ingredients.items():
            self.ingredients[k] = int(v * self.dish_energy_multiplier)

        self.dish_categories = ["カレー", "サラダ", "デザート"]
        # 料理データ（食材の消費量と獲得エナジー）
        with open("dishes_data.json", "r", encoding="utf-8") as f:
            self.all_dishes = json.load(f)

        self.recipe_level_bonuses = [
            1,
            1.02,
            1.04,
            1.06,
            1.08,
            1.09,
            1.11,
            1.13,
            1.16,
            1.18,
            1.19,
            1.21,
            1.23,
            1.24,
            1.26,
            1.28,
            1.30,
            1.31,
            1.33,
            1.35,
            1.37,
            1.40,
            1.42,
            1.45,
            1.47,
            1.50,
            1.52,
            1.55,
            1.58,
            1.61,
            1.64,
            1.67,
            1.70,
            1.74,
            1.77,
            1.81,
            1.84,
            1.88,
            1.92,
            1.96,
            2.00,
            2.04,
            2.08,
            2.13,
            2.17,
            2.22,
            2.27,
            2.32,
            2.37,
            2.42,
            2.48,
            2.53,
            2.59,
            2.65,
            2.71,
            2.77,
            2.83,
            2.90,
            2.97,
            3.03,
            3.09,
            3.15,
            3.21,
            3.27,
            3.34,
        ]

        for dishes in self.all_dishes.values():
            for dish in dishes.values():
                dish["energy"] = int(
                    dish["energy"]
                    * self.dish_energy_multiplier
                    * (
                        self.recipe_level_bonuses[dish["level"] - 1]
                        if self.consider_dish_level
                        else 1
                    )
                )

        self.dishes = self.all_dishes[self.dish_week]
        # 料理の作成決定変数（0 or 1）
        self.cooked_dishes = {}
        for dish in self.dishes:
            for t in self.cook_time:
                self.cooked_dishes[(dish, t)] = self.model.NewBoolVar(
                    f"cooked_{dish}_{t}"
                )

        # 各ポケモンの動員状態をバイナリ変数で表現
        self.pokemon_active = {}
        for pokemon in self.pokemon_data:
            for d in range(self.today, self.days):  # 0時～71時
                self.pokemon_active[(pokemon, d)] = self.model.NewBoolVar(
                    f"active_{pokemon}_{d}"
                )

        self.stamina = {}
        for pokemon in self.pokemon_data:
            for d in range(self.today, self.days):  # 0時～71時
                self.stamina[(pokemon, d)] = self.model.NewIntVar(-999, 999,
                    f"stamina_{pokemon}_{d}"
                )

        self.pokemon_evolution = {}
        for pokemon in self.future_pokemon:
            self.pokemon_evolution[pokemon] = self.model.NewBoolVar(
                f"evolution_{pokemon}"
            )

        self.ingredients_additionals = {}
        for ingredient in self.ingredients.keys():
            for t in self.cook_time:
                self.ingredients_additionals[(ingredient, t)] = self.model.NewIntVar(
                    0, self.storage_limit, f"additional_{ingredient}_{t}"
                )
        for p in self.pokemon_data:
            if "berry_burst" in self.pokemon_data[p]:
                self.pokemon_data[p]["energy_charge"] = int(
                    self.pokemon_data[p]["berry_burst"]
                    * self.pokemon_data[p]["berries"]
                )

        for p in self.pokemon_data:
            if "energy_charge" in self.pokemon_data[p]:
                self.pokemon_data[p]["energy_charge"] = int(
                    self.pokemon_data[p]["energy_charge"] * self.skill_energy_multiplier
                )
        
        self.pokemon_heal_all = {}
        for p in self.pokemon_data:
            if "heal_all" in self.pokemon_data[p]:
                self.pokemon_heal_all[p] = self.pokemon_data[p]

        self.heal_pool = {}
        for d in range(self.today, self.days):
            self.heal_pool[d] = self.model.NewIntVar(0, 999, f"heal_pool_{d}")

        self.heal = {}
        for p in self.pokemon_data:
            for d in range(self.today, self.days):
                self.heal[(p,d)] = self.model.NewIntVar(0,999,f"heal_{p}_{d}")

        waking_time = self.bedin_time-self.day_switch_time
        for p in self.pokemon_data:
            self.pokemon_data[p]["heal_all"] = int(self.pokemon_data[p].get("heal_all", 0) * waking_time)
            self.pokemon_data[p]["heal_yell"] = int(self.pokemon_data[p].get("heal_yell", 0) * waking_time)
            self.pokemon_data[p]["heal_self"] = int(self.pokemon_data[p].get("heal_self", 0) * waking_time)

    def add_constraints(self):
        self.consumed_ingredients = {i: {t: 0 for t in self.cook_time} for i in self.ingredients}

        if not self.use_seed_pokemon:
            for p in self.seed_pokemon:
                for d in range(self.today, self.days):
                    self.model.Add(
                        self.pokemon_active[(p, d)] == 0
                    )

        # 各ポケモンの動員数を進化前のポケモンの進化数以下に制限
        for p in self.pokemon_data:
            for d in range(self.today, self.days):
                self.model.Add(
                    self.pokemon_active[(p, d)] <= self.pokemon_evolution.get(p, 1)
                )

        if not self.use_additional_ingredient:
            for ingredient in self.ingredients.keys():
                for t in self.cook_time:
                    self.model.Add(self.ingredients_additionals[(ingredient, t)] == 0)

        self.model.Add(
            sum(self.pokemon_evolution[p] for p in self.future_pokemon)
            <= self.evolution_num
        )

        for p in self.future_pokemon:
            for d in range(self.today, self.days):
                if "base" in self.future_pokemon[p]:
                    self.model.Add(
                        self.pokemon_active.get((self.future_pokemon[p]["base"], d), 0)
                        <= 1 - self.pokemon_evolution[p]
                    )

        # 各時間帯で最大5匹までのポケモンを動員
        for d in range(self.today, self.days):
            self.model.Add(
                sum(self.pokemon_active[(p, d)] for p in self.pokemon_data)
                <= self.max_active_pokemon
            )
            for p in self.designated_pokemon:
                self.model.Add(self.pokemon_active[(p, d)] == 1)
            for p in self.forbidden_pokemon:
                self.model.Add(self.pokemon_active[(p, d)] == 0)
        for t in self.cook_time:
            self.model.Add(
                sum(self.cooked_dishes[(dish, t)] for dish in self.dishes.keys()) <= 1
            )
        # 各食材の累積所持量を追跡
        self.food_inventory = {}

        for ingredient in self.ingredients:
            self.food_inventory[ingredient] = {}
            for t in range(self.start_time, self.num_hours):
                # 収集量を計算
                collected = []
                for p in self.pokemon_data:
                    collected_var = self.pokemon_active[
                        (p, (t - self.day_switch_time) // 24)
                    ] * self.pokemon_data[p]["ingredients"].get(ingredient, 0)
                    collected.append(collected_var)

                # 前時刻の在庫（t=10 の場合は初期在庫）
                if t == self.start_time:
                    previous_inventory = self.initial_stock.get(ingredient, 0) * 100
                else:
                    previous_inventory = self.food_inventory[ingredient][t - 1]

                # 料理ごとの消費量を変数にする
                # 各料理について食材の消費量をリニアに表現
                if t in self.cook_time:
                    self.consumed_ingredients[ingredient][t] = sum(
                        self.cooked_dishes[(dish, t)]
                        * self.dishes[dish]["ingredients"].get(ingredient, 0)
                        * 100
                        for dish in self.dishes
                    )+ self.ingredients_additionals[(ingredient, t)] * 100

                # 食材の在庫更新
                self.food_inventory[ingredient][t] = (
                    previous_inventory + sum(collected) - self.consumed_ingredients[ingredient].get(t, 0)
                )

                # 在庫が負にならない制約を追加
                self.model.Add(self.food_inventory[ingredient][t] >= 0)

        #for ingredient in self.ingredients:
        #    self.model.Add(
        #        self.food_inventory[ingredient][self.num_hours-1]
        #        >= self.final_stock.get(ingredient, 0) * 100
        #    )

        for t in self.cook_time:
            # 料理の容量制約
            if t // 24 == 6:
                self.model.Add(sum(self.consumed_ingredients[i][t] for i in self.ingredients) <= self.pot_capacity * 200)
            else:
                self.model.Add(sum(self.consumed_ingredients[i][t] for i in self.ingredients) <= self.pot_capacity * 100)

        # 全食材の合計数を制約
        for t in range(self.start_time, self.num_hours):
            self.model.Add(
                sum(
                    self.food_inventory[ingredient][t]
                    for ingredient in self.food_inventory
                )
                <= self.storage_limit * 100
            )

        waking_time = self.bedin_time-self.day_switch_time
        heal_per_day = defaultdict(int)
        for p in self.pokemon_heal_all:
            for d in range(self.today, self.days):
                heal_per_day[d] += self.pokemon_data[p]["heal_all"] * self.pokemon_active[(p,d)]

        for d in range(self.today, self.days):
            self.model.Add(sum(self.heal[(p,d)] for p in self.pokemon_data) <= self.heal_pool[d])
            self.model.Add(self.heal_pool[d] <= sum(self.pokemon_data[p].get("heal_yell", 0)*self.pokemon_active[(p,d)] for p in self.pokemon_data))

        for p in self.pokemon_data:
            for d in range(self.today, self.days):
                heal_ammount = self.pokemon_data[p].get("heal_self", 0) + heal_per_day[d] + self.heal[(p,d)]
                self.model.Add(self.stamina[(p,d)] == 100-waking_time*6*self.pokemon_active[(p,d)] + heal_ammount)
                self.model.Add(self.stamina[(p,d)] >= self.stamina_threshold * self.pokemon_active[(p,d)])

        # 料理によるエナジーの合計を最大化
        self.total_energy = sum(
            self.cooked_dishes[(dish, t)] * self.dishes[dish]["energy"]
            for dish in self.dishes
            for t in self.cook_time
        )
        self.total_energy += sum(
            self.ingredients_additionals[(ingredient, t)] * self.ingredients[ingredient]
            for ingredient in self.ingredients
            for t in self.cook_time
        )

        self.total_energy += sum(
            self.pokemon_active[(p, d)] * self.pokemon_data[p]["berries"] * 24
            for p in self.pokemon_data  # ポケモンリスト
            for d in range(self.today, self.days)  # 0時～71時  # 日数
        )

        self.total_energy += sum(
            (self.pokemon_active[(p, d)] * self.pokemon_data[p].get("energy_charge", 0))
            for p in self.pokemon_data
            for d in range(self.today, self.days)
        )
        self.model.Maximize(self.total_energy)

    def solve(self):
        self.add_constraints()
        self.solver.parameters.num_workers = 1  # CPUコア数に合わせて調整
        self.solver.parameters.log_search_progress = False
        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if status == cp_model.OPTIMAL:
                st.success(f"最適解を発見しました。獲得エナジー: {self.solver.Value(self.total_energy)}")
            elif status == cp_model.FEASIBLE:
                st.success(f"実行可能解を発見しました。獲得エナジー: {self.solver.Value(self.total_energy)}")
            st.subheader("日ごとの編成")
            for d in range(self.today, self.days):
                active_pokemon = [
                    f"{p}({str(self.solver.Value(self.stamina[(p,d)]))})"
                    for p in self.pokemon_data
                    if self.solver.Value(self.pokemon_active[(p, d)])
                ]
                st.text(f"日 {d}: {', '.join(active_pokemon)}")
            st.subheader("\n=== 各時間の食材の所持量 ===")
            for t in self.cook_time:
                inventory_status = {
                    ingredient: self.solver.Value(self.food_inventory[ingredient][t])
                    / 100
                    for ingredient in self.food_inventory
                }
                formatted_inventory = ", ".join(
                    [
                        f"{ingredient}: {amount:.1f}"
                        for ingredient, amount in inventory_status.items()
                    ]
                )
                st.text(
                    f"日 {t//24} 時間 {t%24}: {formatted_inventory}, 合計:{sum(self.solver.Value(self.food_inventory[ingredient][t]) / 100 for ingredient in self.food_inventory):.1f}"
                )

            st.subheader("\n=== 各時間の料理作成状況 ===")
            for t in self.cook_time:
                cooked_dishes = [
                    dish
                    for dish in self.dishes
                    if self.solver.Value(self.cooked_dishes[(dish, t)])
                ]
                if cooked_dishes:
                    st.text(
                        f"日 {t//24} 時間 {t%24}: 作成 {', '.join(cooked_dishes)} (容量 {sum(self.solver.Value(self.consumed_ingredients[i][t]) for i in self.ingredients):.1f})"
                    )
                else:
                    st.text(f"日 {t//24} 時間 {t%24}: 作成なし")
            st.subheader("\n=== 各時間の追加食材の使用量 ===")
            for t in self.cook_time:
                additional_ingredients = {
                    ingredient: self.solver.Value(
                        self.ingredients_additionals[(ingredient, t)]
                    )
                    for ingredient in self.ingredients
                }
                formatted_additionals = ", ".join(
                    [
                        f"{ingredient}: {amount}"
                        for ingredient, amount in additional_ingredients.items()
                        if amount > 0
                    ]
                )
                if formatted_additionals:
                    st.text(f"日 {t//24} 時間 {t%24}: {formatted_additionals}")
                else:
                    st.text(f"日 {t//24} 時間 {t%24}: 追加食材なし")
        else:
            st.error("解が見つかりませんでした。")

# --- メイン処理（UI部分） ---
st.title("Pokemon Sleep 最適編成ソルバー")

# サイドバーで条件設定
with st.sidebar:
    st.header("設定")
    day_mapping = {"月":0, "火":1, "水":2, "木":3, "金":4, "土":5, "日":6}
    selected_day = st.selectbox("現在の曜日", list(day_mapping.keys()))
    max_day = st.selectbox("計算する期間", list(map(lambda x: x+"曜日まで", day_mapping.keys())))
    today_int = day_mapping[selected_day]
    days_int = day_mapping[max_day[0]]+1
    berries_liked = [0,0,0]
    berries = {
            "ほのお": "fire",
            "みず": "water",
            "くさ": "grass",
            "どく": "poison",
            "じめん": "earth",
            "でんき": "electric",
            "こおり": "ice",
            "ドラゴン": "dragon",
            "ゴースト": "ghost",
            "フェアリー": "fairy",
            "ひこう": "bird",
            "ノーマル": "normal",
            "むし": "bug",
            "いわ": "rock",
            "エスパー": "esper",
            "かくとう": "fight",
            "あく": "dark",
            "はがね": "steel",
        }
    for i in range(3):
        berries_liked[i] = st.selectbox(f"カビゴンの好きなきのみ{(i+1)}", list(berries.keys()))
    berries_liked = [berries[k] for k in berries_liked]

    dish_type = st.selectbox("料理タイプ", ["カレー", "サラダ", "デザート"])

    st.subheader("食材在庫")
    input_stock = {}
    # 現在の在庫リストにある食材を入力欄として表示
    default_stock = {
            "ネギ": 0,
            "キノコ": 0,
            "卵": 0,
            "イモ": 0,
            "リンゴ": 0,
            "ハーブ": 0,
            "肉": 0,
            "牛乳": 0,
            "ミツ": 0,
            "オイル": 0,
            "ジンジャー": 0,
            "トマト": 0,
            "カカオ": 0,
            "大豆": 0,
            "コーン": 0,
            "コーヒー": 0,
            "シッポ": 0,
            "カボチャ": 0,
            "アボカド": 0,
        }
    
    for ing, val in default_stock.items():
        input_stock[ing] = st.number_input(f"{ing}", value=val, min_value=0, step=1)

# 計算実行ボタン
if st.button("計算開始"):
    user_inputs = {
        "today": today_int,
        "days": days_int,
        "stock": input_stock,
        "dish": dish_type,
        "berries": berries_liked
    }
    
    with st.spinner("計算中...（数秒〜数十秒かかります）"):
        solver = PokemonDeploymentSolver(user_inputs)
        solver.solve()
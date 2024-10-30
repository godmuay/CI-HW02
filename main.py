import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Step 1: กำหนดตัวแปรอินพุตและเอาต์พุต
market_index = ctrl.Antecedent(np.arange(3000, 5001, 1), 'market_index')
stock_trend = ctrl.Antecedent(np.arange(-10, 11, 1), 'stock_trend')
decision = ctrl.Consequent(np.arange(0, 101, 1), 'decision')

# Step 2: กำหนดฟังก์ชันการเป็นสมาชิก (Membership Functions)
# Market Index
market_index['low'] = fuzz.trimf(market_index.universe, [3000, 3000, 4000])
market_index['medium'] = fuzz.trimf(market_index.universe, [3500, 4000, 4500])
market_index['high'] = fuzz.trimf(market_index.universe, [4000, 5000, 5000])

# Stock Trend
stock_trend['down'] = fuzz.trimf(stock_trend.universe, [-10, -10, 0])
stock_trend['steady'] = fuzz.trimf(stock_trend.universe, [-5, 0, 5])
stock_trend['up'] = fuzz.trimf(stock_trend.universe, [0, 10, 10])

# Decision (0-100, โดย 0 = ขาย, 50 = ถือ, 100 = ซื้อ)
decision['sell'] = fuzz.trimf(decision.universe, [0, 0, 50])
decision['hold'] = fuzz.trimf(decision.universe, [25, 50, 75])
decision['buy'] = fuzz.trimf(decision.universe, [50, 100, 100])

# Step 3: กำหนดกฎของระบบ Fuzzy (Fuzzy Rules)
rule1 = ctrl.Rule(market_index['low'] & stock_trend['down'], decision['sell'])
rule2 = ctrl.Rule(market_index['medium'] & stock_trend['steady'], decision['hold'])
rule3 = ctrl.Rule(market_index['high'] & stock_trend['up'], decision['buy'])
rule4 = ctrl.Rule(market_index['high'] & stock_trend['down'], decision['sell'])
rule5 = ctrl.Rule(market_index['medium'] & stock_trend['up'], decision['buy'])
rule6 = ctrl.Rule(market_index['low'] & stock_trend['steady'], decision['hold'])

# Step 4: สร้างระบบควบคุม Fuzzy Control System
investment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
investment_sim = ctrl.ControlSystemSimulation(investment_ctrl)

# Step 5: การใส่ข้อมูล (Input)
# ตัวอย่าง: ดัชนีตลาด = 4500, แนวโน้มราคาหุ้น = 5%
market_index_value = 4500
stock_trend_value = 5

investment_sim.input['market_index'] = market_index_value
investment_sim.input['stock_trend'] = stock_trend_value

# Step 6: ประมวลผลระบบ Fuzzy
investment_sim.compute()

# Step 7: แสดงผลการตัดสินใจ
decision_result = investment_sim.output['decision']
print(f"Decision Score: {decision_result:.2f}")

# ตรวจสอบค่าเพื่อแสดงผลลัพธ์ที่เหมาะสม
if decision_result < 33:
    recommendation = "Suggestion : Sell"
elif 33 <= decision_result <= 66:
    recommendation = "Suggestion : Hold"
else:
    recommendation = "Suggestion : Buy"

print(recommendation)

# Step 8: วาดกราฟแสดงผลลัพธ์

plt.figure(figsize=(12, 6))

# Market Index Graph
plt.subplot(1, 3, 1)
plt.plot(market_index.universe, market_index['low'].mf, 'b', label='Low')
plt.plot(market_index.universe, market_index['medium'].mf, 'g', label='Medium')
plt.plot(market_index.universe, market_index['high'].mf, 'r', label='High')
plt.axvline(x=market_index_value, color='k', linestyle='--', label=f'Input: {market_index_value}')
plt.title('Market Index')
plt.xlabel('Index')
plt.ylabel('Membership Degree')
plt.legend()

# Stock Trend Graph
plt.subplot(1, 3, 2)
plt.plot(stock_trend.universe, stock_trend['down'].mf, 'b', label='Down')
plt.plot(stock_trend.universe, stock_trend['steady'].mf, 'g', label='Steady')
plt.plot(stock_trend.universe, stock_trend['up'].mf, 'r', label='Up')
plt.axvline(x=stock_trend_value, color='k', linestyle='--', label=f'Input: {stock_trend_value}%')
plt.title('Stock Trend')
plt.xlabel('Trend (%)')
plt.ylabel('Membership Degree')
plt.legend()

# Decision Graph
plt.subplot(1, 3, 3)
plt.plot(decision.universe, decision['sell'].mf, 'b', label='Sell')
plt.plot(decision.universe, decision['hold'].mf, 'g', label='Hold')
plt.plot(decision.universe, decision['buy'].mf, 'r', label='Buy')
plt.axvline(x=decision_result, color='k', linestyle='--', label=f'Result: {decision_result:.2f}')
plt.title('Decision')
plt.xlabel('Decision Score')
plt.ylabel('Membership Degree')
plt.legend()

# เพิ่มข้อความแสดงคำแนะนำในกราฟ Decision
plt.text(decision_result + 2, 0.5, recommendation, fontsize=12, color='black')

plt.tight_layout()
plt.show()

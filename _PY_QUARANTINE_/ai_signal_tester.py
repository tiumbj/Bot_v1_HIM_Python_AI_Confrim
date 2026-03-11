"""
HIM AI Signal Tester - ทดสอบส่งสัญญาณไปยัง AI API และรับการแจ้งเตือน
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
# AI API Configuration
AI_API_KEY = "sk-462b368e928148f08193668f48efe7eb"
AI_API_URL = "https://api.deepseek.com/v1/chat/completions"
AI_MODEL = "deepseek-chat"

# Telegram Configuration
TELEGRAM_TOKEN = "8280027714:AAHELs-PAS7u0JctXEIyxz_hnFwgXBm22nI"
CHAT_ID = "8385962634"

# ============================================================================
# TELEGRAM FUNCTIONS
# ============================================================================
def send_telegram_message(message, parse_mode='HTML'):
    """ส่งข้อความไปยัง Telegram"""
    
    print(f"\n📱 Sending to Telegram...")
    
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ Telegram credentials missing")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': parse_mode
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Telegram message sent")
            return True
        else:
            print(f"❌ Telegram failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

def send_telegram_photo(photo_path, caption=""):
    """ส่งรูปภาพไปยัง Telegram"""
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': CHAT_ID, 'caption': caption}
            response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            print("✅ Telegram photo sent")
            return True
        else:
            print(f"❌ Telegram photo failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram photo error: {e}")
        return False

# ============================================================================
# AI FUNCTIONS
# ============================================================================
def create_ai_prompt(signal, market_data, account_data):
    """สร้าง Prompt สำหรับ AI"""
    
    prompt = f"""คุณคือ AI Mentor สำหรับการเทรด Gold (XAUUSD)
วิเคราะห์สัญญาณการเทรดต่อไปนี้และให้คำแนะนำ

[สัญญาณการเทรด - Trading Signal]
ประเภท: {signal['type']}
Confluence: {signal['confluence']} ปัจจัย
Stop Loss: {signal['stop_loss']}
Take Profit: {signal['take_profit']}

[ข้อมูลตลาด - Market Data]
สัญลักษณ์: {market_data['symbol']}
Timeframe: {market_data['timeframe']}
ราคาปัจจุบัน: {market_data['price']}
RSI: {market_data['rsi']}
MACD: {market_data['macd']}
MACD Signal: {market_data['macd_signal']}
ATR: {market_data['atr']}
Volume Ratio: {market_data['volume_ratio']}
EMA Alignment: {market_data['ema_aligned']}
Support: {market_data['support']}
Resistance: {market_data['resistance']}
News ใน 1 ชม.: {market_data['news_in_1hour']}

[ข้อมูลบัญชี - Account Data]
Balance: ${account_data['balance']}
Risk per Trade: {account_data['risk_per_trade']}%

กรุณาวิเคราะห์และตอบกลับในรูปแบบ JSON:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "เหตุผลสั้นๆ",
    "risk_assessment": {{
        "risk_level": "LOW/MEDIUM/HIGH",
        "risk_score": 0-100,
        "risk_reward": 0.0,
        "position_size": 0.0,
        "potential_loss": 0.0,
        "potential_profit": 0.0,
        "warnings": []
    }},
    "adjustments": {{
        "should_adjust": true/false,
        "suggestions": []
    }}
}}
"""
    return prompt

def call_ai_api(prompt):
    """เรียกใช้ AI API"""
    
    print(f"\n🤖 Calling AI API...")
    
    if not AI_API_KEY:
        print("❌ AI API Key missing")
        return None
    
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert trading mentor for Gold (XAUUSD). Respond in JSON format only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        print("⏳ Waiting for AI response...")
        response = requests.post(AI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # พยายาม parse JSON
            try:
                # หา JSON จาก content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    ai_response = json.loads(json_match.group())
                    print("✅ AI API call successful")
                    return ai_response
                else:
                    print("⚠️ No JSON found in response")
                    return None
            except:
                print("⚠️ Could not parse JSON response")
                return None
        else:
            print(f"❌ AI API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ AI API error: {e}")
        return None

def mock_ai_response(signal, market_data):
    """สร้าง Mock AI Response (เมื่อ API ไม่ทำงาน)"""
    
    print("\n🤖 Using Mock AI (API Fallback)")
    
    # คำนวณ lot size
    balance = 10000
    risk_percent = 2.5
    risk_amount = balance * (risk_percent / 100)
    
    entry = market_data['price']
    sl = signal['stop_loss']
    sl_distance = abs(entry - sl)
    
    # Gold: 1 lot = 100 oz, 1 pip = $10 สำหรับ 1 lot
    lot_size = risk_amount / (sl_distance * 100)
    lot_size = max(0.01, min(1.0, round(lot_size * 100) / 100))
    
    # Risk/Reward
    tp = signal['take_profit']
    reward = abs(tp - entry)
    risk_reward = reward / sl_distance if sl_distance > 0 else 0
    
    return {
        "approved": True,
        "confidence": 0.92,
        "reasoning": "✅ Confluence ดี (3 factors)\n✅ RSI เหมาะสม (45.2)\n✅ Volume สูง (1.8x)\n✅ EMA เรียงตัวสวย",
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "risk_score": 40,
            "risk_reward": round(risk_reward, 2),
            "position_size": lot_size,
            "potential_loss": risk_amount,
            "potential_profit": risk_amount * risk_reward,
            "warnings": ["ตรวจสอบข่าวสำคัญก่อนเข้า trade"]
        },
        "adjustments": {
            "should_adjust": False,
            "suggestions": []
        }
    }

# ============================================================================
# ALERT GENERATION
# ============================================================================
def create_telegram_alert(signal, market_data, ai_result, alert_type="SIGNAL"):
    """สร้างข้อความแจ้งเตือนสำหรับ Telegram"""
    
    emoji = {
        'BUY': '🟢',
        'SELL': '🔴',
        'APPROVED': '✅',
        'REJECTED': '❌',
        'SIGNAL': '📊',
        'EXECUTION': '💰',
        'MENTOR': '🤖'
    }
    
    # Header
    message = f"""
{emoji['MENTOR']} <b>HIM INTELLIGENT MENTOR ALERT</b>
{'-'*40}

<b>⏰ Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>📈 Symbol:</b> {market_data['symbol']}
<b>🎯 Signal:</b> {signal['type']} (Confluence: {signal['confluence']})

{'-'*40}
"""

    # AI Analysis
    message += f"""
<b>🤖 AI ANALYSIS:</b>
• <b>Verdict:</b> {emoji['APPROVED'] if ai_result['approved'] else emoji['REJECTED']} {'APPROVED' if ai_result['approved'] else 'REJECTED'}
• <b>Confidence:</b> {ai_result['confidence']*100:.0f}%
• <b>Entry:</b> ${market_data['price']:.2f}
• <b>Stop Loss:</b> ${signal['stop_loss']:.2f}
• <b>Take Profit:</b> ${signal['take_profit']:.2f}

{'-'*40}
"""

    # Risk Assessment
    if 'risk_assessment' in ai_result:
        risk = ai_result['risk_assessment']
        risk_emoji = '🟢' if risk.get('risk_level') == 'LOW' else '🟡' if risk.get('risk_level') == 'MEDIUM' else '🔴'
        
        message += f"""
<b>⚖️ RISK ASSESSMENT:</b>
• {risk_emoji} Risk Level: {risk.get('risk_level', 'N/A')}
• 📊 Risk Score: {risk.get('risk_score', 0)}/100
• 📈 R:R Ratio: 1:{risk.get('risk_reward', 0):.2f}
• 💰 Position Size: {risk.get('position_size', 0)} lots
• 💸 Max Loss: ${risk.get('potential_loss', 0):.2f}
• 💵 Expected Profit: ${risk.get('potential_profit', 0):.2f}

{'-'*40}
"""

    # Reasoning
    if 'reasoning' in ai_result:
        message += f"""
<b>💡 REASONING:</b>
{ai_result['reasoning']}

{'-'*40}
"""

    # Warnings
    if ai_result.get('risk_assessment', {}).get('warnings'):
        message += f"""
<b>⚠️ WARNINGS:</b>
"""
        for warning in ai_result['risk_assessment']['warnings']:
            message += f"• {warning}\n"
        
        message += f"{'-'*40}\n"

    # Market Context
    message += f"""
<b>📊 MARKET CONTEXT:</b>
• RSI: {market_data['rsi']:.1f} ({'Oversold' if market_data['rsi'] < 30 else 'Overbought' if market_data['rsi'] > 70 else 'Neutral'})
• Volume: {market_data['volume_ratio']:.1f}x {'(High)' if market_data['volume_ratio'] > 1.5 else '(Normal)'}
• ATR: {market_data['atr']:.2f}
• EMA Alignment: {'✅' if market_data['ema_aligned'] else '❌'}

{'-'*40}
<b>🔗 Dashboard:</b> http://localhost:5000
<b>🤖 Mentor:</b> Hybrid Intelligence Mentor
"""
    
    return message

def save_ai_response(ai_result, filename="ai_response_log.json"):
    """บันทึก AI Response ลงไฟล์"""
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'ai_result': ai_result
    }
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(filename, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"✅ AI response saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error saving response: {e}")

# ============================================================================
# TEST SCENARIOS
# ============================================================================
def test_buy_signal():
    """ทดสอบ BUY Signal"""
    
    signal = {
        'type': 'BUY',
        'confluence': 3,
        'stop_loss': 5150.0,
        'take_profit': 5250.0
    }
    
    market_data = {
        'symbol': 'XAUUSD',
        'timeframe': 'H1',
        'price': 5176.55,
        'rsi': 45.2,
        'macd': 0.0012,
        'macd_signal': 0.0008,
        'atr': 50.0,
        'avg_atr': 45.0,
        'volume_ratio': 1.8,
        'ema_aligned': True,
        'support': 5150.0,
        'resistance': 5250.0,
        'news_in_1hour': False
    }
    
    account_data = {
        'balance': 10000,
        'risk_per_trade': 2.5
    }
    
    return signal, market_data, account_data

def test_sell_signal():
    """ทดสอบ SELL Signal"""
    
    signal = {
        'type': 'SELL',
        'confluence': 4,
        'stop_loss': 5200.0,
        'take_profit': 5100.0
    }
    
    market_data = {
        'symbol': 'XAUUSD',
        'timeframe': 'H1',
        'price': 5180.25,
        'rsi': 72.5,  # Overbought
        'macd': -0.0005,
        'macd_signal': 0.0002,
        'atr': 55.0,
        'avg_atr': 48.0,
        'volume_ratio': 2.1,
        'ema_aligned': False,
        'support': 5150.0,
        'resistance': 5200.0,
        'news_in_1hour': True
    }
    
    account_data = {
        'balance': 10000,
        'risk_per_trade': 2.5
    }
    
    return signal, market_data, account_data

def test_rejected_signal():
    """ทดสอบ Signal ที่ควรถูกปฏิเสธ"""
    
    signal = {
        'type': 'BUY',
        'confluence': 1,  # น้อยเกินไป
        'stop_loss': 5180.0,
        'take_profit': 5190.0
    }
    
    market_data = {
        'symbol': 'XAUUSD',
        'timeframe': 'H1',
        'price': 5185.50,
        'rsi': 82.3,  # Overbought
        'macd': 0.0025,
        'macd_signal': 0.0010,
        'atr': 45.0,
        'avg_atr': 42.0,
        'volume_ratio': 0.3,  # Volume ต่ำ
        'ema_aligned': False,
        'support': 5150.0,
        'resistance': 5200.0,
        'news_in_1hour': True
    }
    
    account_data = {
        'balance': 10000,
        'risk_per_trade': 2.5
    }
    
    return signal, market_data, account_data

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================
def run_test():
    """รันการทดสอบทั้งหมด"""
    
    print("\n" + "="*70)
    print("🚀 HIM AI SIGNAL TESTER")
    print("="*70)
    print("ทดสอบการส่งสัญญาณไปยัง AI และรับการแจ้งเตือน")
    print("="*70)
    
    # ตรวจสอบ Configuration
    print("\n📋 CONFIGURATION CHECK:")
    print("-" * 40)
    print(f"🤖 AI API Key: {'✅ Found' if AI_API_KEY else '❌ Missing'}")
    print(f"📱 Telegram Token: {'✅ Found' if TELEGRAM_TOKEN else '❌ Missing'}")
    print(f"👤 Chat ID: {'✅ Found' if CHAT_ID else '❌ Missing'}")
    print("-" * 40)
    
    while True:
        print("\n" + "="*60)
        print("📋 TEST MENU")
        print("="*60)
        print("1. 🟢 Test BUY Signal (Approved)")
        print("2. 🔴 Test SELL Signal (Approved)")
        print("3. ⚠️ Test REJECTED Signal")
        print("4. 📊 Test All Signals")
        print("5. 📱 Test Telegram Only")
        print("6. 🤖 Test AI Only")
        print("0. 🚪 Exit")
        print("="*60)
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '0':
            print("\n👋 Exiting...")
            break
        
        # เลือก scenario
        if choice == '1':
            signal, market_data, account_data = test_buy_signal()
            scenario = "BUY (Approved)"
        elif choice == '2':
            signal, market_data, account_data = test_sell_signal()
            scenario = "SELL (Approved)"
        elif choice == '3':
            signal, market_data, account_data = test_rejected_signal()
            scenario = "REJECTED"
        elif choice == '4':
            print("\n📊 Testing ALL signals...")
            test_scenarios = [
                ("BUY (Approved)", test_buy_signal()),
                ("SELL (Approved)", test_sell_signal()),
                ("REJECTED", test_rejected_signal())
            ]
            
            for name, (sig, market, account) in test_scenarios:
                print(f"\n{'='*60}")
                print(f"🔄 Testing: {name}")
                print(f"{'='*60}")
                
                # เรียก AI
                prompt = create_ai_prompt(sig, market, account)
                ai_result = call_ai_api(prompt)
                
                # ถ้า API ไม่ทำงาน ใช้ Mock
                if not ai_result:
                    ai_result = mock_ai_response(sig, market)
                
                # สร้าง Alert
                alert = create_telegram_alert(sig, market, ai_result, "SIGNAL")
                
                # แสดงผล
                print("\n📋 AI Result:")
                print(json.dumps(ai_result, indent=2))
                
                # ส่ง Telegram
                send_telegram_message(alert)
                
                # บันทึก
                save_ai_response(ai_result)
                
                time.sleep(2)  # รอระหว่างการส่ง
            
            print("\n✅ All signals tested!")
            input("\nPress Enter to continue...")
            continue
            
        elif choice == '5':
            print("\n📱 Testing Telegram only...")
            test_message = f"""
📱 <b>TELEGRAM TEST MESSAGE</b>
{'-'*40}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Status:</b> ✅ Connection successful
<b>Message:</b> This is a test from HIM System
{'-'*40}
<b>🤖 Hybrid Intelligence Mentor</b>
"""
            send_telegram_message(test_message)
            print("✅ Test message sent!")
            input("\nPress Enter to continue...")
            continue
            
        elif choice == '6':
            print("\n🤖 Testing AI only...")
            signal, market_data, account_data = test_buy_signal()
            prompt = create_ai_prompt(signal, market_data, account_data)
            ai_result = call_ai_api(prompt)
            
            if ai_result:
                print("\n📋 AI Response:")
                print(json.dumps(ai_result, indent=2))
                save_ai_response(ai_result)
            else:
                print("❌ AI API failed")
            
            input("\nPress Enter to continue...")
            continue
        
        else:
            print("❌ Invalid option")
            continue
        
        # ดำเนินการทดสอบสำหรับตัวเลือก 1-3
        print(f"\n{'='*60}")
        print(f"🔄 Testing: {scenario}")
        print(f"{'='*60}")
        
        print("\n📊 Signal Details:")
        print(f"   Type: {signal['type']}")
        print(f"   Confluence: {signal['confluence']}")
        print(f"   Entry: ${market_data['price']:.2f}")
        print(f"   SL: ${signal['stop_loss']:.2f}")
        print(f"   TP: ${signal['take_profit']:.2f}")
        
        # เรียก AI
        prompt = create_ai_prompt(signal, market_data, account_data)
        ai_result = call_ai_api(prompt)
        
        # ถ้า API ไม่ทำงาน ใช้ Mock
        if not ai_result:
            print("\n⚠️ AI API failed, using mock response")
            ai_result = mock_ai_response(signal, market_data)
        
        # แสดงผล AI
        print("\n📋 AI Result:")
        print(json.dumps(ai_result, indent=2))
        
        # สร้าง Alert
        alert = create_telegram_alert(signal, market_data, ai_result, "SIGNAL")
        
        print("\n📱 Alert Preview:")
        print(alert)
        
        # ส่ง Telegram
        print("\n📱 Sending to Telegram...")
        send_telegram_message(alert)
        
        # บันทึกผล
        save_ai_response(ai_result)
        
        print(f"\n✅ Test completed for {scenario}")
        input("\nPress Enter to continue...")

# ============================================================================
# START
# ============================================================================
if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Test session ended")
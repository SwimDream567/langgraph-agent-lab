"""天气工具 - 像 Skill 一样开箱即用

原理：
  用 wttr.in 免费 API，不需要注册、不需要 API Key、没有调用次数限制。
  直接传城市名（支持中文），返回格式化的天气信息。

对比：
  和风天气：需要注册 → 申请 Key → 用 LocationID 查询 → 5万次/月免费
  wttr.in：  什么都不用 → 直接查 → 无限制

用法：
  from tools.weather_tool import get_weather

  # 在 LangGraph Agent 里当工具用
  from langchain_core.tools import tool

  # 或者直接调用测试
  result = get_weather("深圳")
  print(result)
"""

import requests
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Query real-time weather for a city (free API, no key required)

    Args:
        city: City name, e.g. "Beijing", "Shenzhen", "Shanghai"
    """
    try:
        # wttr.in 的 JSON 接口
        # lang=zh → 中文天气描述
        # format=j1 → JSON 格式返回
        url = f"https://wttr.in/{city}"
        resp = requests.get(url, params={
            "format": "j1",  # JSON 格式
            "lang": "zh",     # 中文
        }, headers={
            "User-Agent": "WeatherAgent/1.0",  # wttr.in 要求带 UA
        }, timeout=10)

        if resp.status_code != 200:
            return f"天气查询失败（HTTP {resp.status_code}）"

        data = resp.json()

        # 解析当前天气（current_condition 数组的第一个元素）
        current = data["current_condition"][0]

        # 解析天气预报（今天）
        today = data["weather"][0]

        # 最接近的观测站
        area = data.get("nearest_area", [{}])[0]

        return (
            f"城市：{area.get('areaName', [{}])[0].get('value', city)}\n"
            f"地区：{area.get('region', [{}])[0].get('value', '')}，"
            f"{area.get('country', [{}])[0].get('value', '')}\n"
            f"天气：{current.get('lang_zh', [{}])[0].get('value', current.get('weatherDesc', [{}])[0].get('value', ''))}\n"
            f"温度：{current['temp_C']}°C（体感 {current['FeelsLikeC']}°C）\n"
            f"今日温度：{today['mintempC']}°C ~ {today['maxtempC']}°C\n"
            f"风向：{current.get('winddir16Point', '')} {current.get('windspeedKmph', '')}km/h\n"
            f"湿度：{current.get('humidity', '')}%\n"
            f"能见度：{current.get('visibility', '')}km\n"
            f"紫外线指数：{current.get('uvIndex', '')}"
        )

    except requests.exceptions.Timeout:
        return "天气查询超时，请稍后再试"
    except requests.exceptions.ConnectionError:
        return "网络连接失败，请检查网络"
    except KeyError as e:
        return f"天气数据解析失败：缺少字段 {e}"
    except Exception as e:
        return f"天气查询出错：{e}"


# ==========================================
# 测试：直接运行这个文件
# ==========================================
if __name__ == "__main__":
    test_cities = ["深圳", "北京", "上海", "广州", "成都"]
    for city in test_cities:
        print(f"\n{'='*40}")
        print(f"查询：{city}")
        print(f"{'='*40}")
        print(get_weather(city))

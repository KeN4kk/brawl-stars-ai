# Brawl Stars AI Bot 🤖⚔️

Полнофункциональный AI-агент для игры Brawl Stars, который обучается с каждой игрой используя Deep Reinforcement Learning.

## 🎯 Основные возможности

- **Multi-Agent RL**: 8 независимых AI-агентов с разными стратегиями
- **Self-Learning**: Агенты учатся на основе опыта (DQN + PPO)
- **Game Simulation**: Полная симуляция боя в реальном времени
- **Auto-Actions**: Автоматическое движение, стрельба, использование способностей
- **Real-time Visualization**: Визуализация боев и прогресса обучения

## 📁 Структура проекта

```
brawl-stars-ai/
├── agents/              # AI-агенты
├── game/                # Игровая логика и симуляция
├── models/              # Neural Networks (DQN, PPO)
├── training/            # Training pipeline
├── utils/               # Утилиты
├── config/              # Конфигурация
├── requirements.txt     # Зависимости
└── main.py             # Главный скрипт
```

## 🚀 Быстрый старт

```bash
pip install -r requirements.txt
python main.py --train --episodes 1000
```

## 📊 Агенты

1. **Aggressive Brawler** - Атакующая стратегия
2. **Defensive Guardian** - Защитная стратегия
3. **Healer Support** - Поддержка и исцеление
4. **Sniper Scout** - Дальнобойная атака
5. **Tank Protector** - Танк с высокой броней
6. **Speedster** - Быстрый маневренный боец
7. **Balanced Fighter** - Сбалансированная тактика
8. **Intelligent Coordinator** - Координатор команды

Версия: 1.0.0

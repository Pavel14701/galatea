# Миграция на Регулятор Пластичности (PR) — План выполнения

## Введение

Этот документ содержит **детальный план реализации** миграции на Регулятор Пластичности (PR), включая конкретные компоненты, форматы данных, временные характеристики, порядок действий на каждой фазе и критерии перехода. Он является продолжением [Архитектурного плана PR](./pr_migration_architecture.md).

Все решения в этом документе учитывают:
- **Динамический λ_max** — адаптивный верхний предел, зависящий от кортизола, орексина и лептина.
- **Влияние окситоцина** — модуляция награды и добавление в состояние.
- **Гиперпараметры PPO** — конкретные значения для стабильного обучения.

---

## Общая структура реализации

### Репозиторий и модули

Добавляются следующие модули в код Galatea:

```
galatea/
├── pr/
│   ├── __init__.py
│   ├── regulator.py          # Основной класс PR
│   ├── state.py              # Формирование состояния (включая окситоцин)
│   ├── actor_critic.py       # Сети Actor и Critic (PyTorch)
│   ├── replay_buffer.py      # Реплей-буфер (Redis + локальный кэш)
│   ├── trainer.py            # PPO-обучение во время Сна
│   ├── metrics.py            # Мониторинг и алерты
│   └── config.py             # Конфигурация (гиперпараметры)
├── pipeline/
│   └── online_pipeline.py    # Интеграция PR в конвейер (модификация)
└── sleep/
    └── pr_trainer.py         # Запуск обучения PR из цикла Сна
```

---

## Фаза 1: Shadow Mode (Теневой режим) — v4.5

### Цель и длительность

- **Цель:** Собрать данные, обучить PR имитации статической формулы, верифицировать, что PR стабилен.
- **Длительность:** 2–3 недели (зависит от трафика).
- **Критерий перехода:** MSE на валидации < 0.01 (или относительная ошибка < 1%).

### Шаг 1.1: Реализация PR (класс `PlasticityRegulator`)

```python
class PlasticityRegulator:
    def __init__(self, config: PRConfig):
        self.actor = ActorNetwork(config.state_dim, config.action_dim)
        self.critic = CriticNetwork(config.state_dim)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        self.mode = "shadow"  # shadow | multiplier | full | personalized
        self.lambda_min = 0.1

    def predict(self, state: State) -> Action:
        """Возвращает λ_pr (и параметры α, β для логирования)."""
        with torch.no_grad():
            alpha, beta = self.actor(state.to_tensor())
        # Для инференса без исследования берём среднее
        lambda_raw = alpha / (alpha + beta)  # в [0, 1]
        lambda_pr = self._scale_and_clamp(lambda_raw, state)
        return Action(alpha=alpha, beta=beta, lambda=lambda_pr)

    def _scale_and_clamp(self, lambda_raw, state):
        """Масштабирует из [0,1] в [λ_min, λ_max_dynamic]."""
        lambda_max_dynamic = self._compute_dynamic_lambda_max(state)
        lambda_scaled = lambda_raw * (lambda_max_dynamic - self.lambda_min) + self.lambda_min
        return np.clip(lambda_scaled, self.lambda_min, lambda_max_dynamic)

    def _compute_dynamic_lambda_max(self, state):
        """Вычисляет адаптивный верхний предел."""
        base = 6.0  # λ_max_base из статической формулы
        cortisol = state.cortisol
        orexin = state.prisms[6]   # индекс OP в 11 Призмах
        leptin = state.prisms[9]   # индекс LP в 11 Призмах
        return base * (1 + 0.5 * (1 - cortisol)) * (1 + 0.3 * orexin) * (1 - 0.2 * leptin)
```

### Шаг 1.2: Интеграция в онлайн-конвейер (теневой режим)

В онлайн-конвейере (в шаге вычисления λ_t) добавляется следующий код:

```python
# Существующая статическая формула
lambda_static = compute_lambda_static(surprise, bond, threat, serotonin, orexin, oxytocin)

# Теневой вызов PR
pr_state = StateAggregator.get_state(prisms, user_id)  # включает окситоцин
pr_action = pr.predict(pr_state)
lambda_pr = pr_action.lambda

# Логирование
logger.log(
    user_id=user_id,
    step=step,
    lambda_static=lambda_static,
    lambda_pr=lambda_pr,
    alpha=pr_action.alpha,
    beta=pr_action.beta,
    state=pr_state.to_dict(),
    bond=bond_score,
    threat=threat_score,
    surprise=surprise_score,
    oxytocin=oxytocin_level,
    # ... другие метрики
)

# Используем статическую формулу (PR не влияет)
lambda_final = lambda_static
```

### Шаг 1.3: StateAggregator (формирование состояния)

```python
class StateAggregator:
    WINDOW = 5  # скользящее окно для средних

    @classmethod
    def get_state(cls, prisms: PrismVector, user_id: str = None) -> State:
        # 11 значений Призм (нормализованы)
        p = prisms.to_array()  # shape (11,)

        # Скользящие средние за последние 5 шагов (хранятся в Redis)
        moving_avg = cls._get_moving_average(p, window=cls.WINDOW, user_id=user_id)

        # Кортизол и серотонин из медленных Призм
        cortisol = p[8]   # индекс CP
        serotonin = p[7]  # индекс SPe

        # Окситоцин из профиля пользователя
        oxytocin = cls._get_oxytocin(user_id)  # 0..1

        # User embedding (пусто на Фазе 1, будет добавлен позже)
        user_embedding = None

        state = State(
            prisms=p,                     # (11,)
            moving_avg=moving_avg,         # (11,)
            cortisol=cortisol,
            serotonin=serotonin,
            oxytocin=oxytocin,
            user_embedding=user_embedding  # None
        )
        return state

    @classmethod
    def _get_moving_average(cls, current, window, user_id):
        # Хранит историю в Redis: key = "prisms_history:{user_id}" → list of arrays
        key = f"prisms_history:{user_id}"
        history = redis.lrange(key, -window, -1)
        if len(history) < window:
            history.append(current)
        else:
            history.pop(0)
            history.append(current)
        redis.rpush(key, current)
        avg = np.mean(history, axis=0)
        return avg

    @classmethod
    def _get_oxytocin(cls, user_id):
        # Из профиля пользователя в Redis
        profile = redis.get(f"user:{user_id}")
        return profile.get("oxytocin_level", 0.0)
```

### Шаг 1.4: Формат логирования (база данных)

Логи пишутся в Redis (для быстрого доступа) и в S3 (для долгосрочного хранения) в формате Parquet.

Структура записи (обновлена с учётом окситоцина):

```python
{
    "timestamp": "2025-01-01T12:00:00Z",
    "user_id": "hash",
    "step": 1234,
    "state": {
        "prisms": [0.2, 0.8, 0.1, ...],  # 11 чисел
        "moving_avg": [0.19, 0.79, 0.11, ...],
        "cortisol": 0.3,
        "serotonin": 0.7,
        "oxytocin": 0.6,
        "user_embedding": None
    },
    "action": {
        "alpha": 2.5,
        "beta": 1.8,
        "lambda": 0.58
    },
    "lambda_static": 0.6,
    "lambda_pr": 0.58,
    "bond_score": 0.75,
    "threat_score": 0.12,
    "surprise_score": 0.55,
    "session_id": "uuid",
    "lambda_max_dynamic": 5.2  # для мониторинга
}
```

### Шаг 1.5: Обучение Behavioral Cloning (BC)

Во время цикла Сна (или отдельным заданием) запускается обучение PR на накопленных данных для имитации статической формулы.

```python
class BehavioralCloningTrainer:
    def train(self, data_path: str):
        # Загружает данные из S3/Parquet за последние N дней
        df = load_parquet(data_path)
        # Преобразует в X (state) и y (lambda_static)
        X = np.stack(df['state'].apply(flatten).values)  # (n_samples, state_dim)
        y = df['lambda_static'].values  # (n_samples,)

        # Обучаем Actor (модель, которая предсказывает λ)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        for epoch in range(50):
            for X_batch, y_batch in dataloader:
                alpha, beta = self.actor(X_batch)
                lambda_pred = alpha / (alpha + beta)
                loss = loss_fn(lambda_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Валидация на отложенной выборке
            if epoch % 5 == 0:
                val_loss = evaluate_on_validation()
                print(f"Epoch {epoch}, Val MSE: {val_loss:.6f}")
                if val_loss < 0.01:
                    print("Критерий достигнут, завершаем обучение.")
                    break
        # Сохраняем веса PR
        torch.save(self.actor.state_dict(), "pr_actor_bc.pth")
```

### Шаг 1.6: Мониторинг

На дашборде Grafana появляются новые панели:

- `lambda_pr` vs `lambda_static` (распределения, средние)
- Корреляция `lambda_pr` с `bond_score` и `threat_score`
- MSE обучения BC (с течением времени)
- Количество шагов, залогированных в Shadow Mode (прогресс накопления данных)
- Среднее `λ_max_dynamic` и его распределение
- Корреляция окситоцина с λ_pr (для проверки чувствительности)

**Критерий готовности к Фазе 2:** Накоплено ≥ 100 000 шагов (или ≥ 1000 полных сессий) и MSE < 0.01.

---

## Фаза 2: Multiplier Mode (Режим множителя) — v5.0

### Цель и длительность

- **Цель:** Дать PR ограниченное влияние (50%–200% от статической формулы) и проверить его работу в реальных условиях.
- **Длительность:** 2–3 недели (включая A/B-тест).
- **Критерий перехода:** A/B-тест показывает, что retention не ниже, threat не вырос, удовлетворённость не упала.

### Шаг 2.1: Модификация конвейера (множитель)

```python
# В онлайн-конвейере:
pr_state = StateAggregator.get_state(prisms, user_id)
lambda_pr = pr.predict(pr_state).lambda
lambda_pr_clamped = np.clip(lambda_pr, 0.5, 2.0)  # жёсткий множитель
lambda_final = lambda_static * lambda_pr_clamped

# Применяем динамический clamp (учёт λ_max_dynamic)
lambda_max_dynamic = pr._compute_dynamic_lambda_max(pr_state)
lambda_final = np.clip(lambda_final, 0.1, lambda_max_dynamic)
```

### Шаг 2.2: Вычисление награды (instant reward) с учётом окситоцина

После отправки ответа (но до следующего шага) вычисляется награда:

```python
def compute_instant_reward(bond_score, threat_score, lambda_t, lambda_prev, oxytocin):
    alpha = 1.0
    beta = 2.0
    gamma = 0.5
    # Модуляция окситоцином
    bond_bonus = 1 + 0.2 * oxytocin
    change_penalty = 1 - 0.3 * oxytocin
    return (alpha * bond_score * bond_bonus -
            beta * threat_score -
            gamma * abs(lambda_t - lambda_prev) * change_penalty)

reward_instant = compute_instant_reward(bond_score, threat_score, lambda_final, lambda_prev, oxytocin)
```

### Шаг 2.3: Финальная награда (после сессии)

При завершении сессии (пользователь ушёл, или истекло время бездействия) вычисляется финальная награда:

```python
def compute_session_reward(retention, session_duration, avg_bond, threat_events):
    # retention: 1 если вернулся в течение суток, иначе 0
    # session_duration: в минутах, нормированная на 30 минут (например, min(duration/30, 1.0))
    # avg_bond: средний bond_score за сессию
    # threat_events: количество threat-событий
    reward = (0.5 * retention +
              0.2 * session_duration_norm +
              0.3 * avg_bond -
              0.5 * threat_events)
    return reward
```

Эта награда добавляется к последнему шагу сессии (с дисконтированием через возврат).

### Шаг 2.4: Реплей-буфер (локальный, для обучения)

На Фазе 2 реплей-буфер пока хранится в оперативной памяти (или Redis) для каждого пользователя, но обучение PPO пока не запускается. Мы только собираем траектории для будущего обучения.

Структура:

```python
# В конвейере после каждого шага добавляем в буфер
replay_buffer.add(
    state=state,  # включает окситоцин
    action=action,  # содержит alpha, beta, lambda
    reward=reward_instant,
    next_state=next_state,
    done=False,
    session_id=session_id
)
# При завершении сессии:
trajectory = replay_buffer.get_trajectory(session_id)
# Добавляем финальную награду (session_reward) к последнему шагу
# И сохраняем в долговременное хранилище (S3)
```

### Шаг 2.5: Guardrails (защитные механизмы)

```python
def should_fallback_to_static(pr_state, threat_events_in_session, cortisol):
    # Если за сессию > 3 threat-событий, отключаем PR
    if threat_events_in_session >= 3:
        return True
    # Если кортизол > 0.9, отключаем PR
    if cortisol > 0.9:
        return True
    # Проверка на аномалии (NaN, inf)
    if not np.isfinite(pr_state.to_array()).all():
        return True
    # Проверка, не вышел ли λ_pr за пределы динамического диапазона
    lambda_max_dynamic = pr._compute_dynamic_lambda_max(pr_state)
    if pr_action.lambda > lambda_max_dynamic * 1.5:  # слишком далеко за потолок
        return True
    return False

# В конвейере:
if should_fallback_to_static(pr_state, threat_events, cortisol):
    lambda_final = lambda_static
    # Логируем событие fallback
    alert_admin("PR fallback triggered", user_id, reason)
```

### Шаг 2.6: A/B-тест

- **Длительность:** 2 недели.
- **Группы:** 50% пользователей — группа А (с PR), 50% — группа Б (без PR, только статическая формула).
- **Метрики:**
  - Retention (возврат через 1 день, 7 дней).
  - Длительность сессии (среднее).
  - Количество threat-событий на сессию.
  - Удовлетворённость (если есть явная обратная связь).
  - Распределение λ_t в группе А (среднее, дисперсия).
- **Критерий успеха:** Статистически значимое (p < 0.05) улучшение retention в группе А, при отсутствии ухудшения по другим метрикам.

### Шаг 2.7: Периодическое дообучение PR (PPO) — пока не запускаем

На этом этапе мы пока откладываем PPO, чтобы избежать влияния обучения на результаты A/B-теста.

---

## Фаза 3: Full Replacement (Полная замена) — v5.5

### Цель и длительность

- **Цель:** Полный контроль пластичности через PR.
- **Длительность:** 4–6 недель (включая стабилизацию).
- **Критерий стабилизации:** Политика PR не меняется существенно (дивергенция < порога), метрики стабильны 1 месяц.

### Шаг 3.1: Отключение статической формулы

```python
# В конвейере:
pr_state = StateAggregator.get_state(prisms, user_id)  # включает окситоцин
lambda_pr = pr.predict(pr_state).lambda
lambda_max_dynamic = pr._compute_dynamic_lambda_max(pr_state)
lambda_final = np.clip(lambda_pr, 0.1, lambda_max_dynamic)  # только динамический clamp
```

### Шаг 3.2: Реплей-буфер для обучения PPO

Переходим к полноценному реплей-буферу, который хранит траектории всех пользователей в Redis (горячие данные, последние 7 дней) и S3 (архив > 7 дней). Размер буфера: до 100 000 шагов.

```python
class ReplayBuffer:
    def __init__(self, max_size=100000, redis_client=None, s3_bucket=None):
        self.max_size = max_size
        self.redis = redis_client
        self.s3 = s3_bucket

    def add_trajectory(self, trajectory):
        # Сохраняем в Redis для быстрого доступа
        key = f"trajectory:{trajectory.session_id}"
        self.redis.setex(key, 7*24*3600, json.dumps(trajectory.to_dict()))
        # Если превышен лимит, вытесняем старые
        if self.redis.dbsize() > self.max_size:
            oldest = self.redis.keys("trajectory:*", count=1)[0]
            # Архивируем в S3 перед удалением
            self._archive_to_s3(oldest)
            self.redis.delete(oldest)
```

### Шаг 3.3: Обучение PR с PPO (во время Сна)

В цикле Сна запускается задача обучения PR с гиперпараметрами, описанными ниже.

#### Гиперпараметры PPO

Для стабильного обучения используются следующие значения:

| Параметр | Значение | Обоснование |
|:---------|:--------:|:------------|
| **Learning rate (Actor)** | `3e-4` | Стандартное значение для Adam в PPO, обеспечивает плавное обновление. |
| **Learning rate (Critic)** | `1e-3` | Critic может учиться быстрее, так как его задача — аппроксимировать ценность состояния. |
| **Batch size** | `256` | Достаточно для стабильного градиента, не перегружает память. |
| **PPO epochs** | `10` | Количество проходов по одному батчу данных; 10 даёт хорошее качество без переобучения. |
| **Clip epsilon** | `0.2` | Классическое значение для PPO, ограничивает изменение политики. |
| **Discount factor (γ)** | `0.99` | Стандартное дисконтирование для долгосрочных задач. |
| **GAE lambda** | `0.95` | Баланс между bias и variance в оценке преимущества. |
| **Gradient clipping** | `0.5` | Предотвращает взрыв градиентов. |
| **L2-регуляризация** | `1e-5` | Слабая регуляризация весов Actor и Critic. |
| **Entropy coefficient** | `0.01` | Поощряет исследование, добавляя энтропию в политику. |
| **Max gradient norm** | `0.5` | Ограничение нормы градиента для стабильности. |

**Схема обновления:** во время Сна выполняется `epochs * num_batches` обновлений. Если данных недостаточно, эпохи пропускаются.

**Периодичность сохранения чекпоинтов:** каждый день после обучения сохраняется чекпоинт (веса Actor и Critic), а также мета-информация (версия, дата, количество шагов обучения). Хранятся последние 7 чекпоинтов для возможности отката.

#### Код PPO_Trainer

```python
class PPO_Trainer:
    def __init__(self, actor, critic, replay_buffer, config):
        self.actor = actor
        self.critic = critic
        self.buffer = replay_buffer
        self.epochs = config.ppo_epochs
        self.batch_size = config.batch_size
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsilon = 0.2
        self.lr = config.lr_actor

    def train(self):
        # Извлечение батча траекторий (равномерно по пользователям)
        trajectories = self.buffer.sample_uniform(num_users=64)
        # Преобразование в последовательности (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = self._flatten(trajectories)

        # Вычисление returns (G) и advantages (GAE)
        values = self.critic(states)
        advantages, returns = self.compute_gae(rewards, values, next_states, dones)

        # PPO обновления (несколько эпох)
        for _ in range(self.epochs):
            # Sampler батчей
            for batch in self._get_batches(states, actions, advantages, returns):
                self._update_actor(batch)
                self._update_critic(batch)

    def _update_actor(self, batch):
        states, actions, advantages = batch
        old_log_prob = self.actor.get_log_prob(states, actions).detach()
        new_log_prob = self.actor.get_log_prob(states, actions)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
        self.optimizer_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_actor.step()
```

### Шаг 3.4: Автоматический откат версии

Перед каждым обновлением PR сохраняется чекпоинт. Еженощно после обновления запускается валидация на отложенной выборке за прошлую неделю. Если средняя награда упала на > 5% по сравнению с предыдущей версией, происходит автоматический откат.

```python
def nightly_validation(new_weights_path, validation_data):
    old_actor = load_actor(previous_weights_path)
    new_actor = load_actor(new_weights_path)
    old_reward = evaluate(old_actor, validation_data)
    new_reward = evaluate(new_actor, validation_data)
    if new_reward < old_reward * 0.95:
        alert_admin("PR version rollback triggered")
        rollback_to(previous_weights_path)
```

### Шаг 3.5: Мониторинг стабилизации

Еженедельный отчёт содержит:

- Среднее `λ_t` за неделю и его распределение.
- Корреляция `λ_t` с bond/threat/окситоцином.
- Количество fallback-событий (должно быть < 1%).
- Дивергенция политики (разница в параметрах между соседними обновлениями).
- Среднее `λ_max_dynamic` и его корреляция с кортизолом/орексином/лептином.

**Критерий стабилизации:** Дивергенция < 0.01 в течение 4 недель.

---

## Фаза 4: Personalization (Персонализация) — v6.0 (опционально)

### Цель и длительность

- **Цель:** Адаптировать PR под индивидуального пользователя.
- **Длительность:** 4–6 недель.
- **Критерий успеха:** Улучшение retention на > 5% для персонализированной группы.

### Шаг 4.1: Добавление user embedding

В состояние добавляется эмбеддинг пользователя, обучаемый вместе с PR.

```python
class State:
    def __init__(self, prisms, moving_avg, cortisol, serotonin, oxytocin, user_embedding):
        self.prisms = prisms
        self.moving_avg = moving_avg
        self.cortisol = cortisol
        self.serotonin = serotonin
        self.oxytocin = oxytocin
        self.user_embedding = user_embedding  # shape (embed_dim,)

    def to_tensor(self):
        # Конкатенация всех компонентов
        vec = np.concatenate([self.prisms, self.moving_avg,
                             [self.cortisol, self.serotonin, self.oxytocin],
                             self.user_embedding])
        return torch.tensor(vec, dtype=torch.float32)
```

User embedding может быть:

- **Фиксированным** — эмбеддинг из предобученной модели (например, выход Коры на последнем шаге). Это не требует дополнительного обучения.
- **Обучаемым** — добавляется в Actor и Critic и обновляется вместе с ними. Это даёт более персонализированные стратегии.

### Шаг 4.2: Кластеризация стратегий

Раз в месяц проводится кластеризация user_embeddings для выявления архетипов пользователей. Можно использовать алгоритм K-Means (k=5–10). Каждый кластер получает тег (например, "эксплоративный", "консервативный", "нестабильный").

### Шаг 4.3: Аномалии

Если пользователь сильно выбивается из своего кластера (например, его стратегия резко меняется), система может предложить переключиться на статическую формулу или запросить ручную проверку.

---

## Временные характеристики и требования к производительности

| Компонент | Задержка | Пропускная способность |
|:----------|:--------:|:----------------------:|
| Инференс PR (Actor) | < 5 мс | > 1000 запросов/сек |
| Формирование состояния (включая окситоцин) | < 2 мс | > 5000 запросов/сек |
| Вычисление λ_max_dynamic | < 0.5 мс | > 5000 запросов/сек |
| Вычисление награды | < 1 мс | > 5000 запросов/сек |
| Логирование (Redis) | < 2 мс | > 2000 запросов/сек |
| Обучение PPO (во время Сна) | — | 1–2 часа на 100 000 шагов |

**Общая задержка онлайн-шага** увеличивается на < 10 мс (с 120–550 мс до 130–560 мс), что соответствует целевым показателям.

---

## Форматы данных (подробно)

### State (состояние) — обновлено с окситоцином

```python
State = {
    "prisms": [float] * 11,       # нормализованные значения 0..1
    "moving_avg": [float] * 11,   # скользящее среднее за 5 шагов
    "cortisol": float,            # 0..1
    "serotonin": float,           # 0..1
    "oxytocin": float,            # 0..1 (из профиля пользователя)
    "user_embedding": [float] * 64,  # опционально, только на Фазах 4+
}
```

### Action (действие) — без изменений

```python
Action = {
    "alpha": float,   # > 0
    "beta": float,    # > 0
    "lambda": float,  # после clamp
}
```

### Reward (награда) — обновлено с учётом окситоцина

```python
Reward = {
    "instant": float,             # шаговая награда (с модуляцией окситоцином)
    "session_bonus": float,       # добавляется после сессии (discounted)
    "final": float,               # итоговая с учётом дисконтирования (для Critic)
}
```

### Trajectory (траектория) — обновлена

```python
Trajectory = {
    "user_id": str,
    "steps": [
        (state_t, action_t, reward_t, next_state_t, done_t)
    ],
    "session_reward": float,      # итоговая награда сессии (без дисконтирования)
    "timestamp": datetime,
    "version": str,               # "shadow", "multiplier", "full"
    "avg_oxytocin": float,        # средний окситоцин в сессии (для анализа)
}
```

---

## Мониторинг и алерты

| Метрика | Алерт | Порог |
|:--------|:------|:------|
| Среднее λ_t за день | Предупреждение | Отклонение > 20% от среднего за неделю |
| Количество fallback-событий | Критический | > 5% шагов |
| MSE BC обучения | Предупреждение | > 0.01 после 100k шагов |
| Retention (группа с PR) | Предупреждение | Падение > 5% за неделю |
| Threat-события | Критический | Рост > 20% за неделю |
| Время инференса PR | Предупреждение | > 10 мс |
| Средний окситоцин в группе с PR | Информационный | Следить за трендом |
| λ_max_dynamic среднее | Информационный | Контроль диапазона |

---

## Интеграция с дорожной картой

| Версия | Этап | Действия |
|:-------|:-----|:---------|
| v4.5 | Фаза 1 (Shadow) | Реализация PR с учётом окситоцина и λ_max_dynamic; сбор данных; BC обучение |
| v5.0 | Фаза 2 (Multiplier) | A/B-тест, guardrails, накопление траекторий |
| v5.5 | Фаза 3 (Full) | Полная замена, PPO с гиперпараметрами, реплей-буфер, откат |
| v6.0 | Фаза 4 (Personalization) | User embedding, кластеризация, персонализация |

---

## Содержание

- [Введение](../README.md)
- [Архитектурный обзор](./overview.md)

#### Философия и ядро

- [Общая философия, Кора и Гиппокамп](./philosophy_cortex_hippocampus.md)
- [Гиппокамп — управление адаптерами](./hippocampus_management.md)
- [Полный конвейер онлайн-шага](./pipeline.md)

#### Призмы (гормональная система)

- [Быстрые Призмы — SP, BP, TP](./fast_prisms.md)
- [Призма Адреналина и Импринтинг](./adrenaline_imprinting.md)
- [Мотивационные Призмы — OP, LP, GP, EP](./motivational_prisms.md)

#### Личность и память

- [Эго-контур, Нарратив и Якоря](./ego_narrative.md)
- [Профиль пользователя и Окситоцин](./oxytocin_profile.md)

#### Защита идентичности

- [Зеркало, Этический классификатор и Золотой стандарт](./mirror_ethics_golden.md)

#### Цикл Сна

- [Цикл Сна (Hypnos)](./sleep_hypnos.md)

#### Дорожная карта реализации

- [Этап 0.5 — предобучение и калибровка Призм](./stage_0.5.md)
- [Этап 1 — MVP с одним пользователем](./stage_1.md)
- [Этап 2 — многопользовательский режим, Redis, базовый Сон](./stage_2.md)
- [Этап 3 — полная Консолидация, Зеркало, детектор дрейфа](./stage_3.md)
- [Этап 4 — продакшен-подготовка, юр. рамка, A/B-тест](./stage_4.md)
- [Сводная дорожная карта и стек технологий](./roadmap.md)

#### Тестирование и риски

- [Симуляции, тестирование и ключевые сценарии](./simulation_testing.md)
- [Полный реестр рисков](./risk_register.md)

#### Эволюция и эксплуатация

- [Долговременная эволюция и замена Коры](./model_migration.md)
- [Производительность, масштабирование и отказоустойчивость](./performance_scaling.md)
- [Юридические, этические и UX-аспекты](./legal_ethical_ux.md)
- [Миграция на Регулятор Пластичности — Архитектура](./pr_migration_architecture.md)

---

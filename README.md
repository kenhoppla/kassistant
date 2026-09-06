# kassistant

A learning card box in front of your voice assistant.

## The problem

A local LLM in Home Assistant is slow. Not because it is stupid, but because
every request pushes the full list of your entities and their states through the
model as text. On a CPU that easily means 20 seconds for "turn on the light" —
a sentence you say every single evening.

The usual answer to this is: buy a GPU.

## The idea

Most voice commands in a home are repetitions. You do not say something new
every day, you say the same twenty things in slightly different words. That does
not need a language model, it needs a good memory.

kassistant sits in front of your existing agent and keeps index cards:

```
Front:  "Turn on the light in the living room"
Back:   light.turn_on  →  light.living_room_ceiling
```

When a sentence comes in, it is looked up rather than reasoned about. If a card
matches confidently, kassistant runs it directly — that takes milliseconds. If
none does, the request is passed on to your LLM unchanged.

**And kassistant takes notes while it happens.** Every action the LLM triggers is
stored together with the sentence that caused it, as a new card. Next time the
same command takes the fast path.

So the assistant does not just get smarter over time — it gets faster.

## How it works

Sentences are turned into rows of numbers ("embeddings"). Similar sentences get
similar numbers, even when the words differ. Looking one up is then a comparison
of numbers instead of a match against text templates — which is why kassistant
also understands rephrasings that the built-in matcher fails on.

The numbers are computed by **Ollama**, which you are already running for your
LLM. That keeps the integration itself free of heavyweight ML libraries —
important, because Home Assistant OS runs on Alpine Linux, where no suitable
wheels exist for onnxruntime or torch.

Storage is a SQLite file in the config directory. A vector database is
unnecessary at a few thousand cards: a dot product across the whole matrix takes
about a millisecond.

## Requirements

* Home Assistant 2026.1 or newer
* A reachable Ollama instance with an embedding model:
  ```
  ollama pull embeddinggemma
  ```
* An already configured conversation agent (Ollama, OpenAI, Google, …) to
  receive the requests kassistant does not know

## Installation

1. Add this repository as a custom repository in HACS and install **kassistant**
2. Restart Home Assistant
3. *Settings → Devices & Services → Add Integration → kassistant*
4. Enter the Ollama address, the embedding model and the fallback agent
5. *Settings → Voice assistants* → set the conversation agent of your assistant
   to **kassistant**
That is all. kassistant fills its own card box.

## The card box fills itself

On startup, and again whenever your exposed devices change, kassistant takes
Home Assistant's own example sentences — the curated, human-written ones its
built-in matcher uses — and turns them into cards for the devices you have
exposed to Assist. So it knows something on day one instead of after a
fortnight of talking to it.

This only ever *writes* cards. It never acts on them: the agent still obeys the
mode you configured, so an automatic fill cannot cause anything to happen in
your home.

Repeat runs are cheap. Sentences already in the box are dropped before anything
reaches Ollama, so expose a new lamp and only its sentences get embedded.

There is an action, **kassistant: Seed the card box**, for two cases: widening
the sample (`max_per_intent`), and retrying after Ollama was unreachable. If a
fill leaves the box unsearchable — cards stored but no vectors, because the
embedding service was down — Home Assistant shows a repair notice, and the
missing vectors are filled in automatically once Ollama answers again.

## The three modes

kassistant starts out deliberately cautious. Change this under
*Settings → Devices & Services → kassistant → Configure*.

| Mode | What happens |
|---|---|
| **Observe only** (start here) | Everything goes to your LLM. kassistant only collects cards. Nothing can change and nothing can break. |
| **Shadow** | kassistant additionally decides what it *would* have done and logs it. Still nothing is executed. This lets you see how well it matches before arming it. |
| **Active** | Confident matches are executed directly. Everything else still goes to the LLM. |

Leave it in stage 1 for a few days. Without cards there is nothing to look up —
the data has to grow out of the way you actually speak.

## Watching it before you arm it

Two diagnostic sensors show whether it is worth switching to active mode:

| Sensor | What it tells you |
|---|---|
| **kassistant cards** | How much it knows. Attributes split this into `seeded` and `learned`, and show how many are `searchable` — a gap means the embedding service was down while they were stored. |
| **kassistant recognised** | The share of recent requests it recognised confidently. This is the number to watch. |

The second one counts *decisions*, not executions, so it means the same thing in
shadow mode and in active mode — which is the point of being able to watch
before arming anything.

**It stays unknown in observe mode**, because there the router is never asked at
all. Reporting zero would read as "recognised nothing" when the truth is "did
not look". So the sequence is: let it seed itself, switch to **shadow** to start
measuring, then to **active** once the recognised share stops climbing.

Its `average_score` attribute is the other half: compare it with your configured
threshold. Far below means the threshold is out of reach; hovering just under it
means it is set slightly too high.

## Changing the connection later

*Settings → Devices & Services → kassistant → ⋮ → Reconfigure* changes the Ollama
address, the embedding model or the fallback agent without losing the card box.

Changing the embedding model invalidates every stored vector — the cards remain
but have to be embedded again, and the sensor's `searchable` count will show that
until they are.

## When kassistant does nothing

The router may abstain at any time, and that is by design. A wrong but confident
decision is more annoying than a slow one. The request is passed on when:

* no sufficiently similar sentence is known (below the confidence threshold)
* Ollama does not answer
* the stored action cannot be executed

## Why a learned card has to be said twice

A card kassistant learned by watching only starts answering once the same
sentence has produced the same action a second time.

That single rule replaces trying to work out whether a follow-up was a
correction. If the fallback agent gets something wrong once, the card it leaves
behind is filed and never used — it would take the same mistake twice for the
same sentence, which does not happen. Anything you actually say regularly earns
its way in on the second time of asking.

The alternative was guessing from the wording of the next sentence, and that
guess is neither reliable nor language-independent: "turn the kitchen off"
followed by "actually turn it on again" is two valid commands, not a
correction, and the first sentence was never wrong.

Seeded cards are exempt — they come from Home Assistant's curated templates
rather than from watching an agent guess. The `awaiting_confirmation` attribute
on the card sensor shows how many are still warming up.

## Status

Early development. Working: seeding, note taking, the learning loop and lookup.

Still open:

* automatic calibration of the confidence threshold
* a localised confirmation phrase — the fast path currently answers "Ok" in
  every language

Only one instance is supported. All entries would share a single card box file
while keeping separate in-memory indexes, and vectors from two different
embedding models cannot be compared at all.

## Development

Python is provided through [uv](https://docs.astral.sh/uv/) — no system install,
no compiling:

```bash
uv venv --python 3.13
uv pip install numpy pytest aiohttp
uv run pytest tests --ignore=tests/integration -q   # fast tests
uvx ruff check custom_components tests              # linting
uvx ruff format custom_components tests             # formatting
```

There are two test suites.

**Unit tests** (`tests/`) cover the card box, the sentence normalisation and the
vector maths. They deliberately run without Home Assistant installed — that is
also what keeps `store.py`, `text.py` and `embeddings.py` free of HA imports.

**Integration tests** (`tests/integration/`) boot a real Home Assistant and set
the integration up in it. They are what catches API drift when a core release
renames something, and they cover the part that cannot be reasoned about from
the outside: kassistant calling another conversation agent from inside its own
turn. They need more to be installed:

```bash
uv pip install homeassistant pytest-homeassistant-custom-component
# the conversation component pins its own matcher versions
uv pip install "hassil==3.5.0" "home-assistant-intents==2026.1.28"
uv run pytest tests/integration -q
```

## License

Apache-2.0

---
title: GetAround Delay Analysis
emoji: 🚗
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GetAround — Delay Analysis Dashboard

Interactive dashboard that quantifies the impact of late returns on the next
rental, and simulates the effect of a **minimum delay between two rentals** along
with its scope (all cars, or Connect only).

Built with Streamlit and Plotly, served through Docker on Hugging Face Spaces.

**Production URL:** https://alexbarbier-getaround-dashboard.hf.space

## What the dashboard shows

1. **How often cars come back late** — distribution and severity of checkout delays.
2. **What it does to the next driver** — impact rate by check-in type, and the
   cancellation gap that follows.
3. **Threshold simulator** — live trade-off between problems solved and bookings
   blocked.
4. **Reading of the data** — the recommended threshold range, to be A/B tested.

## Folder contents

```
dashboard/
├── app.py           # Streamlit interface (rendering only)
├── utils.py         # data loading and business logic
├── test_utils.py    # 16 tests on the business logic
├── test_app.py      # 7 smoke tests, renders the app without a browser
├── data/            # get_around_delay_analysis.xlsx
├── .streamlit/      # configuration (theme, server options)
├── requirements.txt # Python dependencies
├── Dockerfile       # runtime environment
└── README.md        # this file (its YAML header configures the Space)
```

The `app.py` / `utils.py` split is deliberate: the first only renders, the second
carries every computation. That is what makes the metrics testable independently
of the interface — `test_utils.py` never starts Streamlit.

---

# Running it

Three ways, from the lightest to the closest to production. Steps 1 and 2 are how
you validate before deploying at step 3.

## 1. Locally, without Docker

Requires Python 3.11 and an activated virtual environment.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the browser automatically on **http://localhost:8501**, its default
port.

## 2. Locally, with Docker

Reproduces the Space environment exactly. If this step passes, the deployment will
too.

```bash
cd dashboard
docker build -t getaround-dashboard .
docker run -p 7860:7860 --rm getaround-dashboard
```

Then open **http://localhost:7860**.

> **Why 7860 here and 8501 above?** Hugging Face Spaces expects the application to
> listen on the port declared in the `app_port` YAML header. The `Dockerfile`
> therefore pins that port in its start command:
>
> ```dockerfile
> CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
> ```
>
> `--server.address 0.0.0.0` is required: by default Streamlit only accepts
> connections from inside the container, and the Hugging Face proxy could not
> reach it.
>
> The `-p 7860:7860` maps that container port to the same port on your machine.
> Note that this is also the API container's port. Do not run both with the same
> mapping, or `localhost:7860` serves whichever started first — use `-p 7861:7860`
> for one of them.

## 3. Deploying to Hugging Face Spaces

1. **Create a Space → SDK: `Docker`.**

   The "SDK" tells HF how to run the application. The `Streamlit` mode exists and
   would be enough for a standard case, but `Docker` is used here to control the
   Python version and the dependencies exactly as they are locally.

2. **Push the *contents* of `dashboard/`** to the root of the Space — not the
   folder itself. This README and its YAML header must end up at the root, and the
   dataset must be present under `data/`.

   ```bash
   git clone https://huggingface.co/spaces/AlexBarbier/getaround-dashboard
   cd getaround-dashboard

   git lfs install
   git lfs track "*.xlsx"            # before adding the dataset

   cp -r ../Projet_GetAround/dashboard/. .

   git add -A
   git commit -m "Deploy GetAround delay analysis dashboard"
   git push
   ```

   ```powershell
   git clone https://huggingface.co/spaces/AlexBarbier/getaround-dashboard
   cd getaround-dashboard

   git lfs install
   git lfs track "*.xlsx"

   # -Force on Get-ChildItem so dotfiles and .streamlit/ are included:
   # Copy-Item with a * wildcard skips them silently.
   Get-ChildItem ..\Projet_GetAround\dashboard -Force |
       Copy-Item -Destination . -Recurse -Force

   git add -A
   git commit -m "Deploy GetAround delay analysis dashboard"
   git push
   ```

3. **The Space builds the image and starts the container.** The status turns
   *Building*; the **Logs** tab shows the build output, then Streamlit's.

> The Space logs print a line reading `Local URL: http://localhost:7860`. That
> address is the inside of the container, on Hugging Face's servers — it is not
> reachable from your browser. The public address is the one at the top of this
> README.

## Tests

```bash
cd dashboard
pip install pytest
pytest -q
```

Twenty-three tests in two files. `test_utils.py` checks the business rules against
a hand-built six-row table, small enough that every expected number is verifiable
by reading it — including the properties the recommendation depends on, such as
the trade-off curve being monotonic in both directions. `test_app.py` renders the
app in-process and moves the widgets, which catches a broken f-string or a renamed
column before a deploy does.

---

# Reading of the data

**The lever is real but narrow.** About 9% of rentals follow another rental of the
same car within 12 hours: that is the only population the feature touches. But
inside that subset, late returns clearly raise cancellations — from roughly 11% to
17%.

**Returns diminish fast.** Pushing the threshold up keeps solving cases, but each
extra block of minutes solves fewer of them while it keeps blocking bookings. The
diminishing-returns zone starts around two hours.

**Scope matters more than width.** Lateness concentrates on *mobile* check-ins. A
*Connect-only* rule therefore blocks very few bookings, but leaves most problem
cases untouched. A threshold of **60 to 120 minutes on all cars** covers roughly
half to two-thirds of the problems while blocking under 3% of rentals — a
defensible starting point for an A/B test.

# Known limitations

**Observational analysis.** The figures describe what happened, not what would
happen once a threshold is enforced. A "blocked" booking is not necessarily a lost
rental: the driver may shift their slot or pick another car. The blocked count is
therefore an upper bound on the real cost.

**The chaining window is fixed at 12 hours.** Treating two rentals as consecutive
within that window is a modelling choice, not a property of the data. Varying it
changes the size of the affected population.

**Extreme values.** The delay histogram is clipped to ±5h to stay readable, but the
simulations use raw values, including delays of tens of thousands of minutes that
are almost certainly logging errors.

**Frozen data.** The Excel file is baked into the Docker image. Updating the data
means rebuilding and redeploying the Space.

# Possible extensions

- **Test the sensitivity** of the conclusions to the chaining window and to the
  handling of extreme values.
- **Test the cancellation gap statistically** — 11% against 17% is presented as a
  raw comparison; a two-proportion test over the ~1,700 usable pairs would say
  whether it survives the sample size.
- **Separate cancellations** caused by a late return from those with another cause,
  to tighten the benefit estimate.
- **Segment by city or vehicle type**, since the concentration of back-to-back
  rentals may vary sharply by local market.
- **Decouple the data from the image** by loading it from an external source, so
  the analysis can be refreshed without a redeploy.

---

Source: `get_around_delay_analysis.xlsx`. A rental is *impacted* when the previous
car is returned after its planned departure time; a problem is *solved* when the
enforced delay covers the previous driver's lateness.

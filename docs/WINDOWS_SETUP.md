# Run this on Windows in VS Code

## 1. Put it in the right folder

Extract the zip so you get this exact folder:

```
C:\college\Github\Projects\astera-mission-control
```

Inside it you should see `run_mission.py`, `README.md`, `src/`, `data/`,
and `web/`. If you see a double folder like
`astera-mission-control\astera-mission-control`, move the inner one out.

## 2. Open it in VS Code

File > Open Folder > `C:\college\Github\Projects\astera-mission-control`

## 3. Run the Python mission (no API key needed)

In a VS Code terminal, from that folder:

```powershell
python run_mission.py --incident leo-kv-pressure-01
python run_mission.py --incident leo-kv-pressure-01 --approve
python run_mission.py --incident cosmos-telemetry-gap-03
```

## 4. Run the Next.js dashboard

In a second terminal, from that folder:

```powershell
cd web
npm install
npm run dev
```

Then open http://localhost:3000 in your browser.

Note: `node_modules/` is intentionally NOT in the zip (it is hundreds of MB
and Windows-specific). `npm install` rebuilds it on your machine from
`web/package-lock.json`. That first install takes a couple of minutes.

## 5. Push to GitHub later (optional)

From `C:\college\Github\Projects\astera-mission-control` in GitHub Desktop
or git, create/push it as a new repo under `vishalgwu`. Do not commit
`.env` files — there are none needed for the demo.

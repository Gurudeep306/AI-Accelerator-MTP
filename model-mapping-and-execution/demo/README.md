# Recording the simulation demo

A screen recording of the model-mapping work, run live on your Mac, with your voice over it.
Target length: about 6 to 7 minutes. Everything on screen is real output from the project.

```
demo/
  setup_mac.sh   one-time setup (Python environment, PyTorch, SCALE-Sim, the one-line patch)
  demo.sh        the demo itself: 10 scenes, press Enter to go to the next one
  one_layer.py   runs one layer through SCALE-Sim live and shows every file and number
  results.py     151 layers -> 37 shapes, and the latency of all four exits
```

---

## Step 1. Set up the project (once, the day before)

Open Terminal and go to the project folder:

```bash
cd ~/Desktop/researchPapers/_my_part/model-mapping-and-execution
```

If `python3 --version` says 3.9 or older, install a newer Python first:

```bash
brew install python@3.11
```

Then run the setup (downloads PyTorch and SCALE-Sim, about 10 minutes):

```bash
bash demo/setup_mac.sh
```

It should end with a line like `torch 2.x | scalesim 3.0.0` and `setup done`.

## Step 2. Do one full dry run

```bash
source .venv/bin/activate
bash demo/demo.sh
```

Press Enter after each scene. Check that every scene prints without errors. Scene 6 takes about
15 seconds (it is a real SCALE-Sim run). The first run of each script is slower because Python
loads PyTorch and SCALE-Sim; the second run is the one you record.

To repeat a single scene: `bash demo/demo.sh 6`.

## Step 3. Make the screen easy to read

1. Terminal → Settings → Profiles → Text: font size **18 to 20**. Pick a plain light or dark theme.
2. Make the Terminal window about 120 columns × 32 rows, or full screen (Ctrl+Cmd+F).
3. Shorten the prompt so commands are not pushed off the screen: `export PS1='$ '`
4. Turn on Do Not Disturb, close Slack, mail and browser tabs, hide desktop icons.
5. Plug in a headset or use the MacBook microphone in a quiet room.

## Step 4. Record

**Option A: screen and voice together (simplest).**

1. Press **Cmd + Shift + 5**.
2. Choose **Record Selected Portion** and drag the box over the Terminal window
   (or **Record Entire Screen**).
3. Click **Options** → Microphone → **MacBook Pro Microphone** (or your headset).
4. Click **Record**. Wait 2 seconds, then type `bash demo/demo.sh` and press Enter.
5. Read the voice-over for each scene (Step 5), then press Enter for the next scene.
6. To stop, click the stop button in the menu bar. The video is saved on your Desktop as a `.mov` file.

If you make a mistake, keep going and pause for 2 seconds; you can cut it out later.

**Option B: screen first, voice later (cleaner).** Record the screen without the microphone,
one scene at a time (`bash demo/demo.sh 5`, then `bash demo/demo.sh 6`, ...). Then add your voice
in iMovie with its Voiceover tool (Step 6). Use this if you get nervous speaking while typing.

## Step 5. What to say in each scene

| Scene | On screen | Say (in your own words) | Time |
|---|---|---|---|
| 0 | title and the list of steps | "I am Gurudeep. My part of the project is model mapping and execution: taking the detector and working out how long each of its four exits takes on our 4×4 accelerator. Everything in this video runs live." | 0:30 |
| 1 | the detector files | "These are the detector files exactly as I received them. The network has four forward stages and a detection head after each, so there are four exits." | 0:30 |
| 2 | step 1 trace output | "I run the network once with a hook on every convolution. Each hook records the shape of its layer. The weights and the image don't matter for cycle counts, only shapes do. There are 151 convolutions and 8.888 billion multiply-adds per frame." | 0:45 |
| 3 | PASS | "I check this against the model author's own layer table: 151 layers, 1,208 fields, zero mismatches." | 0:20 |
| 4 | the shapes table | "Many layers repeat, mostly inside the four identical heads. There are only 37 different shapes, and the same shape always takes the same cycles, so I simulate each shape once." | 0:30 |
| 5 | tiny layer: config, topology, report, trace | "Here is a tiny layer through SCALE-Sim. These are the three files it reads: the 4×4 output-stationary array, and one row describing the layer, with the padding added and the channels packed two per cycle for INT8. SCALE-Sim reports 95 cycles. The trace shows the address each array row reads on each cycle; you can see each row start one cycle after the one above. And the formula gives 95 as well." | 1:00 |
| 6 | real layer, about 15 s | "Now a real detector layer, 20 by 20 with 64 channels and 64 filters. SCALE-Sim runs it in about 15 seconds: 470,399 cycles, and the formula gives exactly the same, as does the stored result from the full run. At 200 MHz that is 2.35 milliseconds." | 0:50 |
| 7 | formula check table | "I compared the formula with every SCALE-Sim result I have, on many array sizes and dataflows. Every one is equal to the cycle." | 0:30 |
| 8 | exit latency table | "Each exit adds up the cycles of the layers it runs. At 200 MHz, exit 1 takes 446 milliseconds in INT8 and 233 in INT4, about 2 and 4 frames per second. Exit 4 is the slowest at 618 milliseconds." | 0:40 |
| 9 | summary | "So: 151 layers, 37 shapes, every number checked. Next we measure on the board, measure INT4 accuracy, and give this table to the controller. Thank you." | 0:30 |

Total about 6 minutes 45 seconds.

## Step 6. Edit (iMovie, free on the Mac)

1. Open iMovie → Create New → **Movie** → drag the `.mov` file(s) in.
2. Trim long pauses: click the clip, move the playhead, **Cmd + B** to split, select the part, Delete.
3. Optional title card: **Titles** tab → drag a title onto the start → type the project name and your name.
4. Optional slides: export slides from PowerPoint as images (File → Export → PNG) and drag the
   ones you want between scenes; set each to 4 to 6 seconds.
5. Option B only: click the **microphone icon** under the viewer and record your voice scene by scene.
6. Check the sound level: the voice bars should mostly be green and yellow, not red.
7. **File → Share → File** → Resolution **1080p** → Quality **High** → Next → Save as
   `Gurudeep_CS25M111_Simulation_Demo.mp4`.

## Step 7. Before you submit

- The numbers are readable when the video is played at normal size.
- Your voice is clear and not too fast; nothing important is cut off at the start or end.
- The length is within what the professor asked for.
- The file plays on another computer (send it to your phone and open it).

## If something goes wrong

| Problem | Fix |
|---|---|
| `No module named torch` or `scalesim` | You forgot `source .venv/bin/activate`, or run `bash demo/setup_mac.sh` again. |
| `need Python 3.10 or newer` | `brew install python@3.11`, delete the `.venv` folder, run the setup again. |
| SCALE-Sim crashes with `int(max(...))` | The patch was not applied: run the setup again (it applies the patch). |
| Scene 6 takes much longer than a minute | Close other apps; or record scene 6 separately and cut the waiting in iMovie. |
| Colours look odd in the recording | Use Terminal's Basic profile with a white background. |

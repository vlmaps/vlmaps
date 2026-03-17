"""
Generate PDF report using fpdf2.
"""
from fpdf import FPDF, XPos, YPos
from datetime import date

OUTPUT = "/home/mario/tfg/vlmaps_work_session_report.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY    = (26,  42,  74)
BLUE    = (37,  99, 235)
GRAY    = (107, 114, 128)
LGRAY   = (243, 244, 246)
RED     = (220,  38,  38)
GREEN   = ( 22, 163,  74)
ORANGE  = (234,  88,  12)
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)
YELLOW  = (255, 251, 235)
YLWBRD  = (251, 191,  36)
GRENBG  = (240, 253, 244)
REDBG   = (254, 242, 242)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(25, 25, 25)
        self.set_auto_page_break(True, margin=20)

    # ── Page header / footer ─────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, "VLMaps Work Session Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GRAY)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 10, "Generated automatically -- VLMaps Work Session Report", align="C")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def h1(self, text):
        self.ln(6)
        # coloured left bar
        self.set_fill_color(*NAVY)
        x, y = self.get_x(), self.get_y()
        self.rect(x, y, 3, 8, style="F")
        self.set_x(x + 5)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def h3(self, text):
        self.ln(2)
        self.set_font("Helvetica", "BI", 9.5)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def p(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*BLACK)
        x = self.get_x()
        self.set_x(x + 5)
        self.cell(4, 5.5, chr(149))          # bullet char
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(*LGRAY)
        self.set_text_color(*BLACK)
        self.set_draw_color(*GRAY)
        self.set_line_width(0.2)
        # pad top
        self.ln(1)
        lines = text.split("\n")
        for line in lines:
            self.set_x(self.l_margin)
            self.cell(0, 5, "  " + line, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_font("Helvetica", "", 9.5)

    def note(self, text, bg=None, border=None, label="Note"):
        if bg is None:    bg     = YELLOW
        if border is None: border = YLWBRD
        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.set_line_width(0.5)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*BLACK)
        x0 = self.l_margin
        y0 = self.get_y()
        self.ln(1)
        self.set_x(x0 + 2)
        self.cell(12, 5.5, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5.5, " " + text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def fix(self, text):
        self.note(text, bg=GRENBG, border=GREEN, label="Fix")

    def err(self, text):
        self.note(text, bg=REDBG, border=RED, label="Error")

    def hr(self):
        self.ln(2)
        self.set_draw_color(*GRAY)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def table(self, headers, rows, col_widths):
        # header row
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*BLACK)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(*LGRAY if fill else WHITE)
            for cell, w in zip(row, col_widths):
                # Use multi_cell for wrapping but keep same row height logic
                x0, y0 = self.get_x(), self.get_y()
                self.multi_cell(w, 5.5, cell, border=1, fill=fill,
                                new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
                # advance y to max
            self.ln(5.5)
        self.ln(3)


pdf = PDF()
pdf.add_page()

# ══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
pdf.set_fill_color(*NAVY)
pdf.rect(0, 0, pdf.w, 60, style="F")

pdf.set_y(15)
pdf.set_font("Helvetica", "B", 24)
pdf.set_text_color(*WHITE)
pdf.cell(0, 12, "VLMaps Navigation System", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 13)
pdf.cell(0, 8, "Work Session Report  -  Development & Debugging Log", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(200, 210, 230)
pdf.cell(0, 6, f"Date: {date.today().strftime('%B %d, %Y')}   |   Repository: ~/tfg/vlmaps",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_y(70)
pdf.set_text_color(*BLACK)

# ── Table of contents ─────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(*NAVY)
pdf.cell(0, 8, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_draw_color(*NAVY)
pdf.set_line_width(0.3)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(2)

toc = [
    ("1.", "Project Overview"),
    ("2.", "Task 1 - Generating Evaluation Task Files"),
    ("3.", "Task 2 - OpenAI API Model Migration"),
    ("4.", "Task 3 - Evaluation Pipeline Fixes"),
    ("5.", "Task 4 - Interactive Navigation Demo"),
    ("6.", "Debugging: Why the Robot Was Not Moving"),
    ("7.", "Complete Change Summary"),
    ("8.", "Quick-Start Instructions"),
    ("9.", "Technical Notes and Known Limitations"),
]
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(*BLACK)
for num, title in toc:
    pdf.cell(10, 6, num)
    pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("1.  Project Overview")
pdf.p(
    "This report documents a full development and debugging session on the VLMaps (Vision-Language Maps "
    "for Robot Navigation) repository. VLMaps is a research system that builds a 3-D semantic map of an "
    "indoor environment from a set of RGB-D images and robot poses. At runtime, the map is queried with "
    "natural-language descriptions to plan paths to user-specified objects."
)
pdf.p(
    "The simulator back-end is Habitat-Sim (Meta AI), the scenes come from the HM3D dataset, and "
    "semantic features are extracted with LSeg (Language-driven Semantic Segmentation) and matched "
    "at query time with CLIP (OpenAI). LLM parsing of natural-language instructions uses GPT-4o-mini "
    "via the OpenAI API."
)

pdf.h2("1.1  System Architecture (Three-Stage Pipeline)")
pdf.bullet(
    "OFFLINE MAP BUILDING (create_map.py) -- walks through all RGB-D frames, extracts 512-dim LSeg features, "
    "back-projects depth to 3-D, and accumulates features in a 1000x1000x30 voxel grid saved as vlmap/vlmaps.h5df."
)
pdf.bullet(
    "RUNTIME INITIALISATION (HabitatLanguageRobot.setup_scene()) -- loads the HDF5 map, builds a 2-D "
    "obstacle map by projecting voxels, and runs CLIP against all Matterport-3D categories to produce "
    "a pre-computed score matrix (voxels x categories)."
)
pdf.bullet(
    "NAVIGATION (move_to_object(name)) -- queries the score matrix to get a 2-D semantic mask, "
    "finds the nearest object instance, plans a path with a visibility-graph navigator, and executes it."
)

pdf.h2("1.2  Key Files")
pdf.table(
    ["File / Path", "Role"],
    [
        ["vlmaps/robot/habitat_lang_robot.py",  "Main robot class -- setup, navigation, action execution"],
        ["vlmaps/map/vlmap.py",                  "VLMap class -- load/query 3-D semantic map"],
        ["vlmaps/map/map.py",                    "Base Map -- obstacle map, get_nearest_pos, planner calls"],
        ["vlmaps/navigator/navigator.py",        "Visibility-graph path planner"],
        ["vlmaps/utils/llm_utils.py",            "GPT-4o-mini wrappers for NL instruction parsing"],
        ["vlmaps/utils/index_utils.py",          "CLIP-based category matching (find_similar_category_id)"],
        ["vlmaps/task/habitat_object_nav_task.py","Evaluation task loader and metric tracker"],
        ["application/generate_object_nav_tasks.py","NEW -- generates object_navigation_tasks.json per scene"],
        ["application/interactive_object_nav.py",   "NEW -- interactive NL navigation demo with OpenCV"],
        ["application/evaluation/evaluate_object_goal_navigation.py","Evaluation entry point"],
        ["config/object_goal_navigation_cfg.yaml",  "Hydra config entry point"],
        ["config/params/default.yaml",              "Map params: gs=1000, cs=0.05 m/cell"],
    ],
    [95, 80]
)

pdf.h2("1.3  Pre-built Map for Scene 00800-TEEsavR23oF_2")
pdf.p(
    "The VLMap for this scene was already built before this session. The HDF5 file "
    "(vlmap/vlmaps.h5df, ~143 MB) contains:"
)
pdf.bullet("grid_feat: (14,718 x 512) LSeg feature vectors -- one per occupied voxel")
pdf.bullet("grid_pos:  (14,718 x 3)   3-D voxel coordinates in grid space")
pdf.bullet("grid_rgb:  (14,718 x 3)   RGB colour at each voxel")
pdf.bullet("occupied_ids: (1000 x 1000 x 30)  voxel occupancy grid")
pdf.bullet("weight:    (14,718,)  feature accumulation weights")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- GENERATING TASK FILES
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("2.  Task 1 -- Generating Evaluation Task Files")
pdf.p(
    "The evaluation pipeline requires a file called object_navigation_tasks.json for each scene. "
    "This file defines the robot's starting pose, the target object categories, and scene metadata. "
    "The repository does not ship these files for HM3D scenes, so they had to be generated."
)

pdf.h2("2.1  Script Created: application/generate_object_nav_tasks.py")
pdf.p("The script performs the following steps for the requested scene (scene_id=1):")
pdf.bullet("Reads poses.txt -- a file of 7-D pose vectors [px, py, pz, qx, qy, qz, qw].")
pdf.bullet("Samples N=20 evenly-spaced poses from the full trajectory.")
pdf.bullet("Converts each pose to a 4x4 Habitat transformation matrix via cvt_pose_vec2tf().")
pdf.bullet(
    "Assigns object categories randomly from the Matterport-3D list (mp3dcat), "
    "using a fixed seed (42) for reproducibility."
)
pdf.bullet("Writes a JSON list -- one dict per task, integer-indexed from 0.")

pdf.h2("2.2  JSON Task Schema")
pdf.code("""{
  "task_id": 0,
  "tf_habitat": [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1],  // flat 16-element 4x4 matrix
  "map_grid_size": 1000,
  "map_cell_size": 0.05,
  "scene": "00800-TEEsavR23oF",
  "instruction": "go to the chair",
  "objects_info": [{"name": "chair"}]
}""")
pdf.note(
    "The scene name strips the trailing _N suffix from the folder name "
    "(e.g. 00800-TEEsavR23oF_2 -> 00800-TEEsavR23oF). "
    "The scene index is determined by alphabetical sorting of scene folders."
)

pdf.h2("2.3  Permission Issue on Some Scene Folders")
pdf.err("PermissionError when writing to /home/mario/tfg/data/vlmaps_dataset/00801-HaxA7YrQdEC_*/")
pdf.fix("Run once as sudo:  sudo chown -R mario:mario /home/mario/tfg/data/vlmaps_dataset/00801-*")

pdf.h2("2.4  Running the Script")
pdf.code("""cd ~/tfg/vlmaps
python application/generate_object_nav_tasks.py scene_id=1
# Output: ~/tfg/data/vlmaps_dataset/00800-TEEsavR23oF_2/object_navigation_tasks.json
# Contains 20 tasks with random object categories, reproducible with seed=42""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- OPENAI API MIGRATION
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("3.  Task 2 -- OpenAI API Model Migration")
pdf.p(
    "The repository was originally written for the text-davinci-002 completion endpoint, then partially "
    "updated to gpt-4-turbo. Both models were unavailable on the API key used, causing HTTP 404 errors "
    "at any point where an LLM call was made."
)

pdf.h2("3.1  Error Encountered")
pdf.err("openai.NotFoundError: 404 -- model 'gpt-4-turbo' does not exist on this API key.")

pdf.h2("3.2  Files Modified and Changes Made")
pdf.h3("vlmaps/utils/llm_utils.py")
pdf.p(
    "Both parse_object_goal_instruction() and parse_spatial_instruction() used "
    "model='gpt-4-turbo' in the client.chat.completions.create() call. Changed to model='gpt-4o-mini' "
    "in both functions."
)
pdf.code("""# Before
response = client.chat.completions.create(
    model="gpt-4-turbo",   ...

# After
response = client.chat.completions.create(
    model="gpt-4o-mini",   ...""")

pdf.h3("vlmaps/utils/index_utils.py -- find_similar_category_id()")
pdf.p("Same model change plus a critical response-parsing fix (see §3.3).")

pdf.h2("3.3  GPT-4o-mini Response Parsing Fix")
pdf.p(
    "The original code assumed the model would respond with only the category name. "
    "GPT-4o-mini instead returns a full explanatory sentence, for example:"
)
pdf.code('  "The most relevant category to \'bathtub\' among the list is \'bathtub\'"')
pdf.p(
    "A direct classes_list.index(text) call therefore raised ValueError. "
    "The fix first tries an exact match, then scans the response for any known category name:"
)
pdf.code("""text = response.choices[0].message.content.strip()
if text in classes_list:
    return classes_list.index(text)
# GPT-4o-mini sometimes returns a full sentence
for item in classes_list:
    if item.lower() in text.lower():
        return classes_list.index(item)
print(f"  [warn] Could not parse '{text}', defaulting to index 0")
return 0""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- EVALUATION PIPELINE FIXES
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("4.  Task 3 -- Evaluation Pipeline Fixes")

pdf.h2("4.1  Crash on Missing Semantic Annotations")
pdf.p(
    "When Habitat-Sim loads a scene without its .semantic.glb mesh and info_semantic.json sidecar files, "
    "the semantic scene graph is empty. The evaluation code called get_class_objects(class_name) which "
    "returned an empty list. The subsequent call to np.argsort on an empty array caused:"
)
pdf.err("IndexError: index 0 is out of bounds for axis 0 with size 0  (in find_closest_object_from_class)")
pdf.p("Fix applied to vlmaps/task/habitat_object_nav_task.py:")
pdf.fix("Added an early-exit guard when no objects of the queried class are found on the current floor:")
pdf.code("""if not class_objects:
    print(f"  [warn] No objects of class '{class_name}' found on this floor.")
    return None, float("inf")""")
pdf.p(
    "The evaluation then records distance = infinity for that subgoal (a failure), which is "
    "the correct behaviour when semantic labels are absent."
)

pdf.h2("4.2  Evaluation Script Flow")
pdf.code("""for scene_id in scene_ids:
    robot.setup_scene(scene_id)
    robot.map.init_categories(mp3dcat.copy())    # compute CLIP scores
    object_nav_task.setup_scene(robot.vlmaps_dataloader)
    object_nav_task.load_task()

    for task_id in range(len(object_nav_task.task_dict)):
        object_nav_task.setup_task(task_id)
        object_categories = parse_object_goal_instruction(object_nav_task.instruction)
        robot.set_agent_state(object_nav_task.init_hab_tf)

        for cat in object_categories:
            robot.move_to_object(cat)

        for action in robot.get_recorded_actions():
            object_nav_task.test_step(robot.sim, action)

        object_nav_task.save_single_task_metric(save_path)""")

pdf.h2("4.3  Running the Evaluation")
pdf.code("""cd ~/tfg/vlmaps
python application/evaluation/evaluate_object_goal_navigation.py scene_id=1
# Results -> ~/tfg/data/vlmaps_dataset/00800-TEEsavR23oF_2/vlmap_obj_nav_results/00.json""")
pdf.p(
    "Each result JSON records: task_id, scene, subgoal_success_rate, finished_subgoal_ids, "
    "goal_classes, instruction, and the complete action sequence."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- INTERACTIVE NAVIGATION DEMO
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("5.  Task 4 -- Interactive Navigation Demo")
pdf.p(
    "The goal was an interactive loop where the user types a natural-language instruction, "
    "GPT-4o-mini parses it into a list of object categories, and the robot navigates to each "
    "one in sequence while an OpenCV window shows the camera feed step by step."
)

pdf.h2("5.1  File Created: application/interactive_object_nav.py")
pdf.h3("Plan-then-Replay Architecture")
pdf.p(
    "A naive approach that calls sim.step() inside move_to_object() moves the robot without "
    "any visible rendering -- the user would only see the final position. Instead, the script "
    "separates planning from rendering:"
)
pdf.bullet(
    "PLAN SILENTLY -- call robot.move_to_object(cat). The robot moves internally in the simulator "
    "and records all actions via recorded_actions_list."
)
pdf.bullet("RESET -- restore the robot to the sub-goal's starting pose with set_agent_state(start_tf).")
pdf.bullet(
    "REPLAY -- iterate over the recorded actions, call robot.sim.step(action) and "
    "show_obs() after each step, with an 80 ms delay (~12 fps)."
)
pdf.p(
    "After completing all categories in one instruction, start_tf is updated to the robot's "
    "current position so that subsequent instructions begin from where the robot arrived."
)

pdf.h3("Instruction Parsing (GPT-4o-mini)")
pdf.p(
    "The instruction is sent to GPT-4o-mini with an 8-shot few-shot prompt that maps "
    "natural language to comma-separated category names:"
)
pdf.code(""""go to the chair and then go to another chair"
  ->  "chair, chair"

"navigate to the green sofa, turn right, find chairs, go to the painting"
  ->  "green sofa, chairs, painting\"""")

pdf.h3("OpenCV Visualisation")
pdf.p(
    "A window titled 'Robot view' opens showing the RGB camera feed at each step. "
    "An overlay label displays the current action count and target category."
)

pdf.h2("5.2  Controls and Usage")
pdf.code("""cd ~/tfg/vlmaps
python application/interactive_object_nav.py scene_id=1

Searching for a good starting position...
Best start: pose[25/256]  map=(412,287)  dist_to_obstacle=8.5 cells
Scene: 00800-TEEsavR23oF_2

Enter navigation instruction (or 'quit'): go to the sofa and then find a chair
Parsing instruction...
Targets: ['sofa', 'chair']

Planning path to: sofa
  Path computed: 87 actions. Replaying...
  Done.

Planning path to: chair
  Path computed: 43 actions. Replaying...
  Done.

Instruction complete. Press any key in the window to continue.""")

pdf.bullet("Press any key in the OpenCV window to enter the next instruction.")
pdf.bullet("Type quit / exit / q at the prompt to stop.")
pdf.bullet("cv2.waitKey(80) between frames gives ~12 fps; increase the value to slow down playback.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("6.  Debugging: Why the Robot Was Not Moving")
pdf.p(
    "After the interactive script was first written, the robot consistently produced 0 movement "
    "actions for any queried object. Understanding and fixing this required tracing the complete "
    "navigation data flow."
)

pdf.h2("6.1  Full Navigation Data Flow (move_to_object)")
pdf.code("""robot.move_to_object("sofa")
  |
  +-- _set_nav_curr_pose()
  |     Habitat agent position -> vlmaps_dataloader -> full-map (row, col)
  |
  +-- map.get_nearest_pos(curr_pos, "sofa")
  |     |
  |     +-- index_map("sofa")
  |     |     find_similar_category_id("sofa", mp3dcat)  -> cat_id
  |     |     scores_mat[:, cat_id] > all other scores  -> binary voxel mask
  |     |
  |     +-- pool_3d_label_to_2d(mask, grid_pos)  -> 2-D semantic mask (1000x1000)
  |     +-- crop to [rmin:rmax, cmin:cmax]
  |     +-- morphological closing + Gaussian blur + dilation
  |     +-- get_segment_islands_pos()  -> contours, centers, bbox  (full-map coords)
  |     +-- filter_small_objects(area_thres=10)
  |     +-- select_nearest_obj(centers, curr_pos)  -> nearest instance id
  |     +-- nearest_point_on_polygon(curr_pos, contour)  -> target (row, col)
  |
  +-- move_to(target)
        nav.plan_to(curr_pos, target)  -> list of waypoints
        execute_actions()              -> forward / turn_left / turn_right / stop""")

pdf.h2("6.2  Root Cause: Starting Position Coincides with Object Boundary")
pdf.p(
    "The core issue is that get_nearest_pos returns the nearest point on the target object's "
    "polygon boundary to the robot's current position. When the robot was set to the pose at "
    "75% of the recorded trajectory, that map cell happened to lie exactly on the sofa's 2-D "
    "boundary in the VLMap projection."
)
pdf.p(
    "This is physically expected: the robot's data-collection trajectory walked next to the sofa, "
    "so the sofa's 2-D footprint (projected from 3-D voxels) overlaps with path positions near it."
)
pdf.code("""Robot full-map position    ->  (543, 343)
Nearest sofa boundary point ->  (543, 343)   [same cell!]

nav.plan_to( (543,343), (543,343) )  ->  trivial path, 0 waypoints
execute_actions([])                  ->  only "stop" appended
Recorded actions: ["stop"]           ->  0 movement actions when replaying""")

pdf.h2("6.3  Earlier Attempt: Fixed 75% Index")
pdf.p(
    "The first attempt used base_poses[int(n_poses * 0.75)] as the start. The 75th-percentile "
    "pose happened to be on the sofa boundary. Switching to 10% (base_poses[int(n_poses * 0.1)]) "
    "improved things but was still fragile -- any fixed index can land on furniture."
)

pdf.h2("6.4  Final Fix: Distance-Transform Starting Position")
pdf.p(
    "The definitive fix uses scipy.ndimage.distance_transform_edt on the obstacle map. "
    "For every free cell, the transform computes its Euclidean distance to the nearest obstacle. "
    "Cells deep inside open space (corridors, empty room centres) have large values; cells right "
    "next to walls or furniture have small values."
)
pdf.p(
    "The script samples ~30 evenly-spaced trajectory poses and selects the one whose map cell "
    "has the largest distance value:"
)
pdf.code("""from scipy.ndimage import distance_transform_edt

obs_map  = robot.map.obstacles_map          # True = free, False = obstacle
dist_map = distance_transform_edt(obs_map)  # per-cell distance to nearest obstacle

best_idx, best_dist = 0, -1.0
for i in range(0, n_poses, max(1, n_poses // 30)):
    tf = cvt_pose_vec2tf(poses[i])
    robot.set_agent_state(tf)
    robot._set_nav_curr_pose()
    row, col = int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
    if obs_map[row, col]:                    # confirm free space
        d = float(dist_map[row, col])
        if d > best_dist:
            best_dist, best_idx = d, i

start_tf = cvt_pose_vec2tf(poses[best_idx])""")
pdf.fix(
    "The robot now starts in the centre of open space, guaranteed to be far from all "
    "furniture boundaries, and move_to_object produces non-trivial paths."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- CHANGE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("7.  Complete Change Summary")
pdf.table(
    ["File", "Change", "Reason"],
    [
        ["vlmaps/utils/llm_utils.py",
         "gpt-4-turbo -> gpt-4o-mini in parse_object_goal_instruction() and parse_spatial_instruction()",
         "Model unavailable on current API key (HTTP 404)"],
        ["vlmaps/utils/index_utils.py",
         "gpt-4-turbo -> gpt-4o-mini; added sentence-scan fallback for response parsing",
         "Model 404 + GPT-4o-mini returns full sentences instead of single words"],
        ["vlmaps/task/habitat_object_nav_task.py",
         "Guard in find_closest_object_from_class(): return (None, inf) when class_objects is empty",
         "IndexError crash when semantic sidecar files are missing"],
        ["application/generate_object_nav_tasks.py",
         "NEW FILE -- generates object_navigation_tasks.json for any scene from poses.txt",
         "Required for evaluation; not provided by the repository for HM3D"],
        ["application/interactive_object_nav.py",
         "NEW FILE -- interactive plan-then-replay navigation demo with OpenCV visualisation",
         "No interactive demo existed in the original repository"],
        ["application/interactive_object_nav.py",
         "find_best_start_pose() using distance_transform_edt to select start pose in open space",
         "Fixed 0-action navigation caused by starting position == object boundary"],
    ],
    [50, 75, 50]
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 -- QUICK START
# ══════════════════════════════════════════════════════════════════════════════
pdf.h1("8.  Quick-Start Instructions")

pdf.h2("8.1  Prerequisites")
pdf.bullet("OPENAI_KEY environment variable set to a valid OpenAI API key.")
pdf.bullet("Habitat-Sim installed and HM3D scene meshes present in ~/tfg/data/hm3d.")
pdf.bullet("Pre-built VLMaps dataset in ~/tfg/data/vlmaps_dataset (vlmaps.h5df must exist).")
pdf.bullet("Python env: clip, habitat-sim, hydra-core, omegaconf, opencv-python, scipy, fpdf2.")

pdf.h2("8.2  Generate Task Files (one-time per scene)")
pdf.code("""cd ~/tfg/vlmaps
python application/generate_object_nav_tasks.py scene_id=1
# -> ~/tfg/data/vlmaps_dataset/00800-TEEsavR23oF_2/object_navigation_tasks.json""")

pdf.h2("8.3  Run Evaluation")
pdf.code("""cd ~/tfg/vlmaps
python application/evaluation/evaluate_object_goal_navigation.py scene_id=1
# Results -> <scene_dir>/vlmap_obj_nav_results/NN.json""")

pdf.h2("8.4  Run Interactive Demo")
pdf.code("""cd ~/tfg/vlmaps
python application/interactive_object_nav.py scene_id=1
# OpenCV window opens.  Type instructions at the terminal.
# Examples:
#   go to the sofa
#   navigate to the chair and then find a table
#   go to the bed
# Type 'quit' to exit.""")

pdf.h2("8.5  scene_id Values")
pdf.p(
    "Scene folders in ~/tfg/data/vlmaps_dataset/ are sorted alphabetically and indexed from 0:"
)
pdf.code("""scene_id=0  ->  00800-TEEsavR23oF_1
scene_id=1  ->  00800-TEEsavR23oF_2
...          (more scenes if generate_object_nav_tasks.py has been run for them)""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 -- TECHNICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("9.  Technical Notes and Known Limitations")

pdf.h2("9.1  Coordinate Systems")
pdf.bullet("HABITAT WORLD FRAME -- 3-D (x, y, z) in metres, y-up convention.")
pdf.bullet("FULL MAP FRAME -- 2-D (row, col) in grid cells; origin at map corner. "
           "Conversion via vlmaps_dataloader.to_full_map_pose().")
pdf.bullet("CROPPED MAP FRAME -- 2-D offset by (rmin, cmin); used internally by the Navigator. "
           "All public APIs use full-map coordinates.")
pdf.note(
    "get_pos() in vlmap.py computes contours in cropped-map space then adds back (rmin, cmin) "
    "before returning, so callers always receive full-map coordinates."
)

pdf.h2("9.2  Obstacle Map Convention")
pdf.code("""obstacles_map[r, c] = True   ->  free space  (no voxels project onto this cell)
obstacles_map[r, c] = False  ->  occupied   (at least one voxel in height range h_min..h_max)""")

pdf.h2("9.3  Action Space")
pdf.bullet("move_forward  -- advance forward_dist = 0.1 m.")
pdf.bullet("turn_left / turn_right  -- rotate by turn_angle = 5 degrees.")
pdf.bullet("stop  -- signals subgoal completion; consumed by evaluation metrics, "
           "skipped in the interactive replay loop.")

pdf.h2("9.4  VLMap Quality Dependence")
pdf.p(
    "Navigation quality is upper-bounded by LSeg feature quality at map-building time. "
    "Symptoms of poor features: the robot navigates to the wrong object, or no instances "
    "are found after filter_small_objects (area threshold = 10 cells)."
)

pdf.h2("9.5  Semantic Sidecar Files")
pdf.p(
    "HM3D scenes ship with a .semantic.glb mesh and info_semantic.json that map mesh surface IDs "
    "to object category names. If these are absent or not referenced in the Habitat config, "
    "sim.semantic_scene is empty and all evaluation distances will be infinity."
)

pdf.h2("9.6  Known Remaining Issue")
pdf.p(
    "When the VLMap does not detect any instance of the queried category (e.g. the scene has no "
    "bed, or the LSeg features are too noisy), get_nearest_pos returns curr_pos unchanged. "
    "The robot records only a 'stop' action and the interactive script prints a warning:"
)
pdf.code("  [warn] No path found for 'bed' -- robot may already be at the target or the "
         "category is not in the map. Skipping.")

pdf.output(OUTPUT)
print(f"PDF written to {OUTPUT}")

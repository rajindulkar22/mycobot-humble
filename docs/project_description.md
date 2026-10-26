# Project description

A myCobot 280 Jetson Nano picks a red, black, or white bottle cap and
is meant to set it down on the sticky note that names that color. The
notes may be moved. Nothing in the program stores a fixed drop
coordinate for a color.

This note is the record of the goal, what was achieved, and which
calibration methods earned a place in the running cell. The operator
steps live in the README.

## Goal

Start from a bare arm on a Jetson Nano and reach a guarded pick-and-place
cell with three properties:

1. The arm moves only through checked commands, and it stops before a
   joint limit.
2. A cap can sit anywhere in the calibrated patch of the table. The
   overview camera finds it. The wrist camera finishes the grasp,
   because the gripper hides the cap from the overview on the way down.
3. After the grasp, the cap goes to the note whose handwritten word
   matches the cap. Shuffling the notes must not require a new program.

A second black cap, a new color, and a printed tag on the note were
all considered and left out. The task is one cap of each of the three
trained colors, and one handwritten word per note.

## What was achieved

| Result | Evidence |
| --- | --- |
| The arm can be commanded, stopped, and released without a second process sharing the serial port | Phase 1 services, physical stop and timeout tests |
| A laptop RViz model follows the real joints | Phase 2, same `/joint_states` stream |
| MoveIt can plan a collision-free approach and will not execute it | Phase 3, execution flags left false |
| The overview detector finds one red, one black, and one white cap | YOLO on the live camera, CPU, 640 px |
| The arm closes on a cap it can see | Live closes with jaw values in the hold band, including a red close at 53 and a black close at 50 |
| The wrist camera can pull the gripper onto the cap during the descent | Live red and black descents that finished on the cap after nudges |
| The laptop reads RED, WHITE, and BLACK from separate note crops, including upside-down writing | Live log: three reads, then a drop command for the matching word |
| A coordinate move is refused or halted when a joint would pass the safe margin | Inverse-kinematics check before the move, live stop during it |

The last step of the original goal is not met. The cell knows which
note matches the cap, and it will drive toward the middle of that
yellow rectangle, but that point is only as good as the calibration
coverage. On the tested table the notes sat below the clicked region.
The red cap was released near the calculated point and missed the
paper. The black note projected past X = 280 mm, so the arm refused
the move and the failed-pick cleanup opened the gripper at the park.
Reading the word and placing on the paper are separate problems. The
first is working. The second needs the notes to lie inside a
calibration that was taken on the note row, or a wrist correction on
the way down to the note. That wrist correction is not built.

## How the work was split

The Jetson Nano has about 4 GB of RAM and runs ROS 2 Humble in Docker.
It owns the serial port, both cameras, the detector, and the pick
page. The laptop owns the vision-language model. A 2B Qwen model in a
Q4 file is about 1.6 GB and needs a GPU. The Nano cannot host it.

That split is also why the laptop is not asked where a note is. A
single photo of the whole table came back with the three colors in
the wrong left-to-right order. The overview camera already has the
order, because it has the rectangles. The laptop is asked one cropped
note at a time and may answer only RED, WHITE, BLACK, or UNREADABLE.

MoveIt stayed a planning tool. The physical picks go through the
hardware coordinate service in short steps. A 419-point planned
trajectory was never a reason to turn execution on. A plan that
reaches a marker does not say the pads will meet the cap.

## Calibration approaches

Every approach answers a different question. Mixing them up is how a
correct word still misses the paper.

```mermaid
flowchart TD
  pixel[Pixel in the overview]
  card[Card homography]
  ruler[Ruler transform into joint1]
  click[Click or ArUco affine into the tool pose]
  wrist[Wrist nudge during descent]
  pixel --> card
  card --> ruler
  pixel --> click
  click --> wrist
  card -.->|display and table markers| rviz[RViz]
  ruler -.-> rviz
  click -->|pick and drop command| arm[Arm XY]
  wrist -->|last centimetres of a pick| arm
```

### 1. Ruler transform from the arm base to the table

`table_transform.yaml` is a measured translation and rotation from
`joint1` to `table_frame`. It was checked by eye in RViz. The table
surface used there sits at Z = −33.4 mm in the base frame, and the
table's own +Z points down.

This was the right tool for drawing caps and a tabletop in RViz, and
for the early approach-planning experiments. It is the wrong tool for
a grasp. It does not know where the camera pixels are, and it does not
know where the gripping pads are relative to the tool point. The file
itself says it needed physical validation before grasping. That
validation never made this the pick mapping.

### 2. Credit-card homography

An ISO ID-1 card, 85.60 × 53.98 mm, was laid on the table. The four
corners were marked in the rectified overview image, and a projective
homography turned pixels into metres on the card's plane. The four
corners reconstruct the card exactly, which any four-point homography
will do. `table_localizer` still uses this file to publish
`/overview_camera/table_detections` for the preview.

It is a poor pick mapping for three reasons.

- The card covered roughly a 100 × 60 pixel patch. A cap or a note
  outside that patch is an extrapolation, and the error grows with
  distance from the card.
- The result is a point on the table plane. The arm is commanded in
  tool coordinates. The pads are not the tool origin, so a perfect
  table point can still miss the cap by the gripper offset.
- The homography belongs to one camera pose. Move the camera or the
  park pose and the metres move with it.

The card file is kept for the preview. The pick node does not command
those metres.

### 3. Clicked pixels paired with the live tool pose

This is the mapping the arm uses. On the calibrate page you click a
mark in the overview, put the centre between the open pads on that
same mark, and capture. The stored pair is the pixel and the tool
pose at that moment. Five pairs in different parts of the image are
fit with an affine model: pixel `(u, v)` plus a constant maps to
robot `(x, y)`. A sixth click is held out as a check and must land
within 12 mm. Each fit spot must leave a residual under 8 mm, and the
spots must span at least 100 px across and 80 px down. An earlier set
put three spots on one row. The fit looked tidy and the check click
was about 25 mm off, because a line does not constrain both image
axes.

On the accepted calibration the five residuals are about 2 to 6 mm.
That is good enough to start a pick inside the clicked patch. It is
not a promise outside it. The accepted clicks sit around pixel rows
90 to 190. The sticky notes on the tested table sat near row 314.
Projecting that row through the fit is a guess. The red note happened
to fall inside the reach box and the arm opened there, off the paper.
The black note, further to the right, fell past X = 280 mm and was
refused.

This method is the best of the ones that were tried, because the
captured pose already includes the pad offset and the arm's own
coordinate frame. Its limit is coverage. Spots have to be taken where
the caps and the notes actually sit, at the same park pose the camera
uses later.

### 4. Automatic pad centre, and an ArUco marker, as ways to collect those pairs

Two helpers feed the same affine fit. Neither replaces it.

Auto-capture looks for the white gripper pads in the overview and
saves that pixel with the tool pose, so there is nothing to click.
It loses the pads when they are small, clipped, or confused with the
arm. The page tells you to use the marker instead when that happens.

ArUco capture uses a printed `DICT_4X4_50` marker, id 0. The marker is
taped flat, the pad centre is placed on the marker centre, and the
detected centre is stored with the tool pose. The pixel is more
repeatable than a mouse click. The marker is moved to each new spot.
The print page is `/aruco`.

ArUco was also the obvious way to label the drop spots. A tag does
not depend on handwriting, and its centre is stable. It was not used
for drops. The task was to read the written word and follow the note
when the notes are shuffled. A tag would have solved identity more
reliably. It would not, by itself, have fixed a calibration that does
not cover the note row. The handwritten path was kept, with the
laptop naming each crop.

### 5. Fixed drop coordinates

The calibration file still has `drop_red`, `drop_black`, and
`drop_white` millimetre spots, saved by parking the tool on a place
and capturing. They are correct only while those places stay put.
The place path does not read them. A shuffled note would get the old
spot.

### 6. Wrist scale, measured rather than taken from the overview fit

The overview fit cannot be reused on the wrist image. The wrist lens
moves with the tool, and its axes are rotated relative to the robot
axes. The scale was measured at travel height by stepping the tool
10 mm over a cap and recording how the cap moved in the image. One
early sign choice walked the arm away from the cap: an error of 89 px
grew to 343 px. The signs in use now pull the cap toward the yellow
cross, and the gain is reduced as the camera gets closer to the table.

The green circle is allowed to stay on a detection only when it is
the same disc: within 28 px and a similar radius. Earlier, looser
rules locked onto the floor, a cable, a hat, the red ring of the
gripper, and a reddish patch of wood. Those were rejected by requiring
a cap-sized circle, a saturated red or a dark disc, and the pale or
red logo in the middle. The wrist correction runs on the pick. It does
not run on the drop, which is why a bad note coordinate is not cleaned
up on the way down.

## What is worth keeping

These are the choices the live cell depends on. The others stay in
the tree as previews or as rejected alternatives.

- Command the arm through the guarded hardware services. Keep MoveIt
  from executing.
- Find the cap with YOLO on the overview, one box per color.
- Turn that pixel into a tool pose with the five-spot affine fit, and
  retake the spots if the camera or the note row moves.
- Finish the grasp with the wrist camera. Leave wrist guidance on.
- Name each note with one laptop crop. Do not ask the model to list
  the table.
- Refuse an unreadable note, a duplicate word, and a point outside
  X 50–280 mm or Y −200–200 mm.
- Check inverse kinematics before a coordinate move, and stop if a
  live joint reaches the 2° margin inside the joint limit.

## What is worth doing next

The open gap is the drop, not the reader.

- Take the five fit spots on the row where the notes sit, plus a
  check click on a note, and accept again. Until the check click on
  a note is inside 12 mm, the arm should not be trusted to open there.
- Keep the notes inside X = 280 mm. A black note to the right of a
  red note already at about 254 mm is outside the box, and the
  current failure path then opens the gripper at the park.
- Add the same wrist recentring on the yellow note before the gripper
  opens. That is the correction that made the picks land. The drop
  does not have it yet.
- Do not open the gripper when the place was refused. The cap should
  stay in the pads until a point inside the box is actually reached.

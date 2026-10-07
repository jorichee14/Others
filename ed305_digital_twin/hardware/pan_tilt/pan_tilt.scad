// ED305 camera pan-tilt mount, 3D printed, set by hand and locked.
//
// Pan:  a pan plate turns on a base disc about an M6 bolt; a pointer on the
//       plate reads the degree scale engraved on the base (L/R from front).
//       Locked by tightening the M6 knob.
// Tilt: the camera cradle turns between the two yoke arms on M5 axle bolts;
//       an M4 thumbscrew through an arc slot in the right arm clamps it. A
//       keyed pointer washer under the thumbscrew reads the tilt scale
//       engraved on that arm (degrees below horizontal).
// The tilt axis crosses the pan axis inside the camera body, so turning
// is close to a pure rotation about the camera (what measure_rotation.py
// assumes).
//
// Frame: pan axis = z, camera looks along +y at pan 0 / tilt 0, the right arm
// (scale, lock) is on +x. Units mm.
//
// Render one part:  openscad -D 'part="yoke"' -o yoke.stl pan_tilt.scad
// part = "base" | "yoke" | "cradle" | "washer" | "knob" | "wall_bracket" | "assembly"

part = "assembly";

// ---- camera: CHECK against the Basler a2A1920-51gc drawing and your lens ----
cam_w = 29;            // housing width (x)
cam_h = 29;            // housing height (z)
cam_len = 42;          // housing length without lens (y)
lens_d = 34;           // lens outer diameter
lens_len = 30;         // lens length in front of the housing
axis_from_front = 15;  // tilt axis behind the lens flange (housing front face)
// camera fixing through the cradle floor (positions relative to the tilt axis, y forward)
cam_m3_holes = [[0, -6], [0, 6]];   // PLACEHOLDER: set from the drawing
tripod_hole = true;                 // 1/4"-20 clearance at y = -axis_from_front + cam_len/2 (Basler tripod adapter)

// ---- ranges ----
ceiling = false;       // true: base screwed to the ceiling, upside down
tilt_min = -10;        // degrees below horizontal, in the room
tilt_max = 90;

// ---- hardware ----
m3_clear = 3.4;  m4_clear = 4.5;  m5_clear = 5.4;  m6_clear = 6.5;  q20_clear = 6.8;
m4_insert_d = 5.6; m4_insert_l = 6;   // heat-set inserts (CNC Kitchen style sizes)
m5_insert_d = 6.4; m5_insert_l = 7;
m6_head_af = 10.2; m6_head_h = 4.4;   // hex head pocket in the base
m6_nut_af = 10.2;  m6_nut_h = 5.2;    // hex nut pocket in the knob

// ---- structure ----
base_t = 8;
plate_r = 27; plate_t = 6;   // pan plate: this disc plus the arms' footprint
arm_t = 8;  cheek_t = 7;  gap = 0.4;  cradle_t = 5;   // cheek_t >= insert lengths
r_lock = 18;                          // lock screw radius from the tilt axis
slot_w = 4.4;
washer_r = 5.5;
eng = 0.6;                            // engraving depth
$fn = 72;

cheek_in  = max(cam_w, lens_d) / 2 + 1;
cheek_out = cheek_in + cheek_t;
arm_in    = cheek_out + gap;
arm_out   = arm_in + arm_t;
H         = axis_from_front + lens_len + 14;  // tilt axis above the plate top: lens clears the knob at 90 deg
zA        = plate_t + H;                       // tilt axis height above the base top
R_cheek   = r_lock + washer_r + 2;
R_arm     = r_lock + washer_r + 15;            // room for ticks and numbers
arm_w     = 46;
plate_reach = sqrt(arm_out * arm_out + arm_w * arm_w / 4);   // pan plate corner radius
scale_r   = plate_reach + 1.5;                 // pan ticks start just outside the plate
base_r    = scale_r + 15;

// mount tilt: + = front goes toward the base. On the ceiling that is up in the room.
function mount_tilt(t) = ceiling ? -t : t;
mt_lo = min(mount_tilt(tilt_min), mount_tilt(tilt_max));
mt_hi = max(mount_tilt(tilt_min), mount_tilt(tilt_max));
// lock screw at angle 180 (behind the axis) at mount tilt 0; at mount tilt t it sits at 180 - t
function lock_angle(mt) = 180 - mt;

module yz(x0, thickness) {   // 2D in (y, z) -> slab from x0 to x0 + thickness
    translate([x0, 0, 0]) rotate([90, 0, 90]) linear_extrude(thickness) children();
}

module arc_slot(r, w, a0, a1, step = 2) {   // slot of width w along radius r, angle a0..a1 (2D)
    for (a = [a0 : step : a1 - step / 2]) hull() {
        translate([r * cos(a), r * sin(a)]) circle(d = w, $fn = 24);
        translate([r * cos(min(a + step, a1)), r * sin(min(a + step, a1))]) circle(d = w, $fn = 24);
    }
}

// ------------------------------------------------------------------ base
module base() {
    difference() {
        translate([0, 0, -base_t]) cylinder(r = base_r, h = base_t);
        // pivot: M6 bolt from below, head captured
        translate([0, 0, -base_t - 1]) cylinder(d = m6_clear, h = base_t + 2);
        translate([0, 0, -base_t - 0.01]) cylinder(d = m6_head_af / cos(30), h = m6_head_h, $fn = 6);
        // fixing: 4x M4 countersunk (under the pan plate, fit before it)
        for (a = [45 : 90 : 315]) rotate(a) translate([19, 0, -base_t - 1]) {
            cylinder(d = m4_clear, h = base_t + 2);
            translate([0, 0, base_t + 1 - 2.5]) cylinder(d1 = m4_clear, d2 = 8.6, h = 2.51);
        }
        // pan scale: 0 at front (+y), L counter-clockwise from above
        for (d = [-180 : 2 : 178]) rotate(d) {
            L = d % 30 == 0 ? 7 : (d % 10 == 0 ? 5 : 3);
            translate([-0.3, scale_r, -eng]) cube([0.6, L, eng + 0.01]);
        }
        for (d = [-150 : 30 : 180]) rotate(d) translate([0, scale_r + 11, -eng])
            linear_extrude(eng + 0.01) rotate(0) text(d == 0 ? "0" : (abs(d) == 180 ? "180" : str(d > 0 ? "L" : "R", abs(d))),
                                                      size = 3.4, halign = "center", valign = "center", font = "Liberation Sans:style=Bold");
    }
}

// ------------------------------------------------------------------ yoke (pan plate + arms)
module arm_profile() {
    hull() {
        translate([0, zA]) circle(r = R_arm);
        translate([-arm_w / 2, plate_t - 0.01]) square([arm_w, 1]);
    }
}

module yoke() {
    difference() {
        union() {
            // pan plate: covers the arms' footprint (no overhang), pointer at the front
            linear_extrude(plate_t) {
                hull() { circle(r = plate_r); translate([-arm_out, -arm_w / 2]) square([2 * arm_out, arm_w]); }
                polygon([[-5, plate_r - 3], [5, plate_r - 3], [0.6, scale_r + 4], [-0.6, scale_r + 4]]);
            }
            // arms
            for (s = [-1, 1]) yz(s > 0 ? arm_in : -arm_out, arm_t) arm_profile();
        }
        // pivot hole
        translate([0, 0, -1]) cylinder(d = m6_clear, h = plate_t + 2);
        // tilt axle holes, both arms
        translate([-arm_out - 1, 0, zA]) rotate([0, 90, 0]) cylinder(d = m5_clear, h = 2 * arm_out + 2);
        // lock slot, right arm
        yz(arm_in - 1, arm_t + 2) translate([0, zA]) arc_slot(r_lock, slot_w, lock_angle(mt_hi), lock_angle(mt_lo));
        // tilt scale on the right arm's outer face: labels in room degrees below horizontal
        yz(arm_out - eng, eng + 0.01) translate([0, zA]) {
            for (t = [ceil(tilt_min / 5) * 5 : 5 : tilt_max]) rotate(lock_angle(mount_tilt(t))) {
                L = t % 30 == 0 ? 6 : (t % 10 == 0 ? 4.5 : 3);
                translate([r_lock + washer_r + 1.5, -0.3]) square([L, 0.6]);
            }
            for (t = [0 : 30 : tilt_max]) rotate(lock_angle(mount_tilt(t)))
                translate([r_lock + washer_r + 11, 0]) rotate(-lock_angle(mount_tilt(t)))
                    text(str(t), size = 3.4, halign = "center", valign = "center", font = "Liberation Sans:style=Bold");
        }
        // left arm: which way is down (helps on the ceiling)
        yz(-arm_out - 0.01, eng + 0.01) translate([0, zA - R_arm + 8]) mirror([1, 0])
            text(ceiling ? "CEILING" : "WALL", size = 3.2, halign = "center", font = "Liberation Sans:style=Bold");
    }
}

// ------------------------------------------------------------------ cradle (in its own frame: tilt axis at origin)
cam_y0 = -(cam_len - axis_from_front);   // housing rear
cam_y1 = axis_from_front;                // housing front (lens flange)
floor_z = -cam_h / 2;

module cheek_profile() {
    hull() {
        circle(r = R_cheek);
        translate([cam_y0, floor_z - cradle_t]) square([cam_y1 - cam_y0, cradle_t]);
    }
}

module cradle() {
    difference() {
        union() {
            // floor under the housing
            translate([-cheek_in, cam_y0, floor_z - cradle_t]) cube([2 * cheek_in, cam_y1 - cam_y0, cradle_t]);
            for (s = [-1, 1]) yz(s > 0 ? cheek_in : -cheek_out, cheek_t) cheek_profile();
        }
        // axle inserts, M5, from both outer faces
        for (s = [-1, 1]) translate([s > 0 ? cheek_out - m5_insert_l : -cheek_out - 0.01, 0, 0])
            rotate([0, 90, 0]) cylinder(d = m5_insert_d, h = m5_insert_l + 0.01);
        // lock insert, M4, right cheek, behind the axis
        translate([cheek_out - m4_insert_l, -r_lock, 0]) rotate([0, 90, 0]) cylinder(d = m4_insert_d, h = m4_insert_l + 0.01);
        // camera fixing
        for (p = cam_m3_holes) translate([p[0], p[1], floor_z - cradle_t - 1]) {
            cylinder(d = m3_clear, h = cradle_t + 2);
            cylinder(d = 6.2, h = 1 + 2.2);                                  // head counterbore
        }
        if (tripod_hole) translate([0, cam_y0 + cam_len / 2, floor_z - cradle_t - 1]) {
            cylinder(d = q20_clear, h = cradle_t + 2);
            cylinder(d = 11.5, h = 1 + 2.2);
        }
        // front edge mark: aligns the housing front with the axis offset
        translate([-cheek_in - 0.01, cam_y1 - 0.3, floor_z]) cube([2 * cheek_in + 0.02, 0.6, 0.6]);
    }
}

// ------------------------------------------------------------------ pointer washer (keyed to the slot)
module washer() {
    difference() {
        union() {
            cylinder(r = washer_r, h = 2.4);
            // needle: radial, points away from the tilt axis
            linear_extrude(2.4) polygon([[-washer_r + 0.5, -2.2], [-washer_r + 0.5, 2.2], [washer_r + 3.5, 0.3], [washer_r + 3.5, -0.3]]);
            // key: rides in the slot, keeps the needle radial
            translate([-0.01, 0, 0]) mirror([0, 0, 1]) translate([-3, -slot_w / 2 + 0.25, 0]) cube([6, slot_w - 0.5, 1.6]);
        }
        translate([0, 0, -2]) cylinder(d = m4_clear, h = 6);
    }
}
// needle along +x of the washer, key along y: the slot runs tangentially, the needle radially

// ------------------------------------------------------------------ M6 knob
module knob() {
    difference() {
        union() {
            cylinder(r = 12, h = 9);
            for (a = [0 : 30 : 330]) rotate(a) translate([12, 0, 0]) cylinder(r = 2.2, h = 9, $fn = 24);
        }
        translate([0, 0, -1]) cylinder(d = m6_clear, h = 11);
        translate([0, 0, 9 - m6_nut_h]) cylinder(d = m6_nut_af / cos(30), h = m6_nut_h + 0.01, $fn = 6);
    }
}

// ------------------------------------------------------------------ wall bracket (shelf for the base, hangs below it)
shelf_d = base_r + 6;      // pan axis to wall
module wall_bracket() {
    w = 2 * base_r;
    difference() {
        union() {
            translate([-w / 2, -shelf_d, -base_t - 6]) cube([w, shelf_d + base_r, 6]);           // shelf
            translate([-w / 2, -shelf_d, -base_t - 6 - 70]) cube([w, 6, 70]);                    // wall plate
            for (x = [-w / 2, w / 2 - 6])                                                         // gussets
                translate([x, -shelf_d + 5.99, -base_t - 6]) rotate([0, 90, 0]) linear_extrude(6)
                    polygon([[0, 0], [64, 0], [0, 40]]);
        }
        // base fixing: M4 through, nut below
        for (a = [45 : 90 : 315]) rotate(a) translate([19, 0, -base_t - 8]) cylinder(d = m4_clear, h = 10);
        // clearance for the M6 head pocket side (head sits in the base)
        // wall fixing: 4 countersunk holes
        for (x = [-w / 2 + 16, w / 2 - 16]) for (z = [-base_t - 6 - 20, -base_t - 6 - 58])
            translate([x, -shelf_d - 1, z]) rotate([-90, 0, 0]) {
                cylinder(d = 4.6, h = 8);
                translate([0, 0, 1 + 6 - 3]) cylinder(d1 = 4.6, d2 = 9.5, h = 3.01);
            }
    }
}

// ------------------------------------------------------------------ interference check
// part="check": what the moving cradle + camera share with the yoke + knob at
// tilt check_tilt (room degrees). Empty output = no collision.
check_tilt = 0;
check_pan = 0;   // part="check_wall": everything above the base, panned, against the wall bracket
module moving(t) translate([0, 0, zA]) rotate([-mount_tilt(t), 0, 0]) { cradle(); camera_dummy(); }

// ------------------------------------------------------------------ assembly preview
pan_preview = 25;
tilt_preview = 30;
module camera_dummy() {
    color("#3a3a3a") translate([-cam_w / 2, cam_y0, floor_z]) cube([cam_w, cam_len, cam_h]);
    color("#222") translate([0, cam_y1, 0]) rotate([-90, 0, 0]) cylinder(d = lens_d, h = lens_len);
}

if (part == "base") base();
else if (part == "yoke") yoke();
else if (part == "cradle") cradle();
else if (part == "washer") washer();
else if (part == "knob") knob();
else if (part == "wall_bracket") wall_bracket();
else if (part == "check") intersection() { moving(check_tilt); union() { yoke(); translate([0, 0, plate_t]) knob(); } }
else if (part == "check_wall") intersection() { wall_bracket(); rotate(check_pan) { yoke(); moving(check_tilt); } }
else if (part == "assembly") {
    color("#9aa7b4") wall_bracket();
    color("#d9d6cf") base();
    rotate(pan_preview) {
        color("#2a78d6") yoke();
        color("#e8a33d") translate([0, 0, plate_t]) knob();
        translate([0, 0, zA]) rotate([-mount_tilt(tilt_preview), 0, 0]) {
            color("#86b6ef") cradle();
            camera_dummy();
        }
        // pointer washer on the right arm at the lock screw
        a = lock_angle(mount_tilt(tilt_preview));
        color("#d03b3b") translate([arm_out, r_lock * cos(a), zA + r_lock * sin(a)])
            rotate([90, 0, 90]) rotate(a) washer();
    }
}

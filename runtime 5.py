# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out

#F = ma               (force is mass times acceleration)
#dv/dt = a            (acceleration is the rate of change in velocity per second)
#dx/dt = v            (velocity is the rate of change in position per second)


import sys, pygame

from math import radians
from math import pi



from pygame.locals import *
from pygame.constants import *
from pygame.font import Font

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL import GLUT
 
from gameobjects.matrix44 import *
from gameobjects.vector3 import *

import random

#from OpenGL.GLUT import 


# IMPORT OBJECT LOADER
from objloader import *
 
pygame.init()
viewport = (1200,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
 
glLightfv(GL_LIGHT0, GL_POSITION,  (0, 200, 0, 1.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))


glEnable(GL_LIGHT0)

glLightfv(GL_LIGHT0+1, GL_POSITION,  (0, 0, 0, 1.0))
#glLightfv(GL_LIGHT0+1, GL_AMBIENT, (0.0, 0.5, 0.0, 1.0))
glLightfv(GL_LIGHT0+1, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0+1, GL_SPECULAR, (0.5, 1.0, 0.5, 1.0))

glEnable(GL_LIGHT0+1)



glEnable(GL_POINT_SMOOTH)
glEnable(GL_PROGRAM_POINT_SIZE)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded
 
# LOAD OBJECT AFTER PYGAME INIT

obj = OBJ('untitled.obj', swapyz=True)
clock = pygame.time.Clock()




glMatrixMode(GL_PROJECTION)
glLoadIdentity()



glFrustum (-1, 1, -1, 1, 3.0, 10.0);

gluLookAt (0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

width, height = viewport
gluPerspective(35, width/float(height), 1, 100.0)

"""
GLdouble eyeX,
GLdouble eyeY,
GLdouble eyeZ,
GLdouble centerX,
GLdouble centerY,
GLdouble centerZ,
GLdouble upX,
GLdouble upY,
GLdouble upZ);
"""




# glEnable(GL_DEPTH_TEST)


glMatrixMode(GL_MODELVIEW)







model_world_radius = 32.0
camera_distance = 10.0

# Camera transform matrix
camera_matrix = Matrix44()
camera_matrix.translate = (0, 0, -model_world_radius+camera_distance)

camera_static = Matrix44()


# User transform matrix
user_matrix = Matrix44()
user_matrix.translate = (0, 0, -model_world_radius)

# User transform matrix
ship_matrix = Matrix44()
ship_matrix.translate = (0.0, 0.0, -model_world_radius)

ass_matrix = Matrix44()
ass_matrix.translate = (0.0,0.0,0.0)

# Initialize speeds and directions
inertial_direction = Vector3()

inertial_direction.set(0,0,0)

rotation_cumul = Vector3(0.0,0.0,0.0)

rotation_direction = Vector3()

rotation_matrix_trace = Matrix44()

rotation_matrix_trace.translate = (0, 0, 1)

rotation_speed = radians(90.0)

movement_position = Vector3()

movement_direction = Vector3()

movement_speed = 10.0    

rs = 1 #rotation increment
ms = 1 #motion increment

height_in_sector = 0
 
rx, ry = (0,0)
tx, ty = (0,0)
zpos = 0
rotate = move = False
pressed = {}

polytimer = 0
polytimer_max = len(obj.vertices)
root_timer = 0

current_sector = 0

alpha = float()
beta = float()
gamma = float()

presscounter = 0

trace_counter = 0
#frames between traces
trace_counter_max = 4    
#length of trace array
trace_length_max = 200

print(str(len(obj.faces))+' faces')
print(str(len(obj.normals))+' normals')
print(str(len(obj.vertices))+' vertices')


canonical_surface_marker = Vector3()
lockvector = Vector3()





class Trace:
    """visual trace"""
    def draw(self, env_mat):
        """draws visible tail/trace of positions"""
        t_list = glGenLists(2)
        glNewList(t_list, GL_COMPILE)
        glBegin(GL_LINES)
        glColor3f(0.9, 0.9, 0.9)
        oxy = 1
        for zv in self.vertices:
            if oxy < len(self.vertices):
                v = self.vertices[oxy]
                p = self.vertices[oxy-1]
                glVertex3f(v[0], v[1], v[2])
                glVertex3f(p[0], p[1], p[2])
                oxy += 1
        glEnd()
        glEndList()
        return t_list
    
    def __init__(self):
        self.vertices = []
        self.append = [(0.0, 0.0, 0.0)]
    

TRACE = Trace()        
POLY_CENTERS_ARRAY = {}


def build_polygon_topography(gon_obj, method, amt):
    """form_the_obj set up the object's topographical distortion.
    from Class OBJ-Loader:
    self.vertices = []
    self.normals = []
    self.texcoords = []
    self.faces = []
    for vertices in face normal = normalize(cross(v1-v0, and v2-v0))
    step 1: move the vertices how we want.
    step 2: recalculate the face normals.
    """

    if method is 'nodal_random':
        heightmax = 0.25

        radius = (model_world_radius*pi*2)/(amt)

        selections = []
        print(radius)
        for e in range(amt):
            selections.append(random.randrange(0, len(gon_obj.vertices)))
        print(selections)

        offsetmax = 1
        offsetmin = 2

        for vertexid in selections:
            vS = Vector3(gon_obj.vertices[vertexid])
            for vertex in gon_obj.vertices:
                v0 = Vector3(vertex)
                d = vS.get_distance_to(v0)
                if d < radius and d > 0:
                    ed = (radius/d)
                    if ed > offsetmax: offsetmax = ed
                    if ed < offsetmin: offsetmin = ed

        #print(offsetmin,offsetmax)

        for vertexid in selections:
            vS = Vector3(gon_obj.vertices[vertexid])
            index = 0
            for vertex in gon_obj.vertices:
                v0 = Vector3(vertex)
                d = vS.get_distance_to(v0)

                if d < radius:
                    if d > 0:
                        ed = (radius / d)
                        nv = ((ed - offsetmin) / (offsetmax - offsetmin)) * heightmax
                    elif d == 0:
                        nv = heightmax

                    pvertex = (v0 * (1 + nv))

                    gon_obj.vertices[index] = (pvertex.x, pvertex.y, pvertex.z)

                index += 1




    if method is 'arbitrary_random':
        scale_offset_max = amt
        index = 0
        for vertex in gon_obj.vertices:
            v0 = Vector3(vertex)
            ct = random.random() * scale_offset_max
            vertex = (v0*(1-ct))
            gon_obj.vertices[index] = (vertex[0], vertex[1], vertex[2])
            index += 1
        pass




    index = 0
    for face in gon_obj.faces:
        vertices, normals, texture_coords, material = face
        #print(face)
        v0 = Vector3((gon_obj.vertices[vertices[0]-1]))
        v1 = Vector3((gon_obj.vertices[vertices[1]-1]))
        v2 = Vector3((gon_obj.vertices[vertices[2]-1]))
        vA = (v1-v0)
        vB = (v2-v0)
        vC = vA.cross(vB)
        vC.normalise()

        gon_obj.normals[index] = (vC.x, vC.y, vC.z)
        index += 1
        #///print(face)

    #return gon_obj
    pass


def build_polygon_centers(gon_obj):
    """construct center-point vertices for all faces.
    populate POLY_CENTERS_ARRAY dictionary in form: k:main_name_int, v:vector
    prints some metrics"""

    index = 0
    for face in gon_obj.faces:
        vertices, normals, texture_coords, material = face
        v1 = Vector3()
        for ii in range(len(vertices)):
            vg = Vector3((gon_obj.vertices[vertices[ii] - 1]))
            v1 += vg
        #heading = Vector3(camera_matrix.right)
        v1 /= 3
        POLY_CENTERS_ARRAY[index] = v1
        index += 1

    fl = len(POLY_CENTERS_ARRAY)
    print("POLY_CENTERS_ARRAY", fl, POLY_CENTERS_ARRAY[fl - 1])
    pass



# build_polygon_topography(obj, 'nodal_random', 12)
# build_polygon_topography(obj, 'arbitrary_random', 0.01)
# obj.refresh()
build_polygon_centers(obj)


def paint_a_gon(gon_obj, index):
    """used to indicate the current sector"""
    gl_list = glGenLists(2)
    glNewList(gl_list, GL_COMPILE)
    glFrontFace(GL_CCW)
    
    vertices, normals, texture_coords, material = gon_obj.faces[index]
    glColor4f(0.8, 0.1, 0.8, 0.1)
    
    glBegin(GL_POLYGON)
    for i in range(len(vertices)):
        glVertex3fv(gon_obj.vertices[vertices[i] - 1])

    glEnd()
    glEndList()
        
    return gl_list


def draw_view_box():
    """for camera testing"""
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)

    glPointSize = 80.0
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)

    for vec in bounds.front:
        pt = bounds.front[vec] #*bounds.pl[plane]['n']
        glVertex3f(10.0, 10.0, 10.0)
        glVertex3f(pt.x,pt.y,pt.z)

    glEnd()
    glEndList()

    return a_gl_list


def draw_camera_view(camera_mat,ass_mat):

    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)

    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 0.1)
    glVertex3f(0.0, 0.0, 0.0)

    t = camera_static.up
    v = Vector3(t[0], t[1], t[2])
    glVertex3f(v.x, v.y, v.z)

    glVertex3f(0.0, 0.0, 0.0)

    t = camera_static.right
    v = Vector3(t[0], t[1], t[2])
    glVertex3f(v.x, v.y, v.z)

    glEnd()

    glEndList()



    return u_gl_list


def draw_user(atmat):
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)

    if box.clipped_state == 0:
        glColor4f(1.0, 0.0, 0.0, 0.4)
    else:
        glColor4f(0.0, 1.0, 0.1, 0.4)

    t = atmat.translate
    v = Vector3(t[0],t[1],t[2])
    v = v*1

    glTranslatef(v.x,v.y,v.z)
    glutWireCube(1.0)

    glEndList()

    return u_gl_list


def draw_ship(atmat):
    s_gl_list = glGenLists(1)
    glNewList(s_gl_list, GL_COMPILE)

    glPushMatrix()
    glScale(0.2, 0.2, 0.2)
    glutSolidIcosahedron()
    glPopMatrix()

    dl = 32
    o = 0.2
        
    bx = abs(atmat.x*dl)/2
    by = abs(atmat.y*dl)/2
    bz = abs(atmat.z*dl)/2

    if atmat.x<0:bx *= -1
    posx = Vector3((o*bx), 0.0, 0.0)
    draw_pologon_marker(posx, abs(atmat.x*dl))
    
    if atmat.y<0:by *= -1
    posy = Vector3(0.0, (o*by), 0.0)
    draw_pologon_marker(posy, abs(atmat.y*dl))
    
    if atmat.z<0:bz *= -1
    posz = Vector3(0.0, 0.0,(o*bz))
    draw_pologon_marker(posz, abs(atmat.z*dl))

    glEndList()
    return s_gl_list


def draw_anchor(atmat):
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)
    
    t = atmat.translate
    v = Vector3(t[0],t[1],t[2])
    v = v.__mul__(1)

    glBegin(GL_LINES)
    glColor3f(0.5,0.5,0.5)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(-v.x,-v.y,-v.z)
    glEnd()
    glEndList()

    return a_gl_list


def draw_reg_mark(vector):
    r_gl_list = glGenLists(1)
    glNewList(r_gl_list, GL_COMPILE)
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(vector.x,vector.y,vector.z)
    glEnd()
    glEndList()
    return r_gl_list


def debug(chars):
    for v in str(chars):
        glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN, ord(v))


def draw_line(atmat, dpos, plength):
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)

    t = atmat.translate
    v = Vector3(t[0], t[1], t[2])
    v = v.__mul__(1)

    glPointSize = 40.0

    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)

    canonical_surface_marker.set(-20.0, 0.0, -40.0)

    glVertex3f(dpos.x, dpos.y, dpos.z)
    glVertex3f(v.x, v.y, v.z)
    glEnd()
    glEndList()

    return a_gl_list


def draw_pologon_marker(pos, plength):
    vg = pos.copy()
    vgn = vg.normalise()
    cve = Vector3((pos[0]+vgn.x), (pos[1]+vgn.y), (pos[2]+vgn.z))
    v1 = Vector3(0.0, 0.0, -1.0)
    v2 = vgn.copy()
    
    a_dot = v1.dot(v2)
    angle = acos(a_dot)
    axis = v1.cross(v2)
    axis = axis.normalise()
    
    l = plength*0.1
 
    glPushMatrix()
    glColor4f(1.0, 1.0, 0.1, 0.1)
    glTranslatef(cve.x, cve.y, cve.z)
    glRotate(degrees(angle), axis.x, axis.y, axis.z)
    glutSolidCone(0.04, l, 12, 12)
    glPopMatrix()


def get_vertex_id(at_vector, scope_normalized, c_sector):
    vertexcount = 0
    vert_set = []
    cs = c_sector
    get_vertex_h = 0
    get_poly_norm = Vector3()
    avc = at_vector.copy().normalize()

    for kk in POLY_CENTERS_ARRAY:
        v = POLY_CENTERS_ARRAY[kk]
        v.normalize()
        d = avc.get_distance_to(v)
        if d < scope_normalized:
            vert_set.append(vertexcount)
        vertexcount += 1
    
    for v in vert_set:
        traf = Vector3()
        dve = obj.faces[v][0]
        va = Vector3(obj.vertices[dve[0]-1])
        vb = Vector3(obj.vertices[dve[1]-1])
        vc = Vector3(obj.vertices[dve[2]-1])
        d = intersect_test(traf, at_vector, va, vb, vc)

        if d: 
            get_poly_norm = d[1]
            get_vertex_h = d[0]
            cs = int(v)

    return [cs, get_vertex_h, get_poly_norm]


def intersect_test(p,d,V0,V1,V2):
    """http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/"""

    e1 = V1.__sub__(V0)
    e2 = V2.__sub__(V0)
    h = d.cross(e2)
    a = e1.dot(h)
    
    if a > -0.00001 and a < 0.00001: return False
    
    f = 1/a
    s = p.__sub__(V0)
    u = f * s.dot(h)
    
    if u < 0.0 or u > 1.0: return False
    
    q = s.cross(e1)
    v = f * d.dot(q)

    if v < 0.0 or u + v > 1.0: return False

    t = f * e2.dot(q)
    n = e1.cross(e2)
    
    if (abs(t) > 0.00001):
        return (t, n)
    else:
        return False






class viewClipBox:
    def __init__(self, near_depth_dist, far_depth_dist, vwidth, vheight):
        print("new viewClipBox")
        self.fpl = {'lt':Vector3(),
                    'rt':Vector3(),
                    'rb': Vector3(),
                    'lb': Vector3(),
                    'fc': Vector3(),
                    'fr': Vector3(),
                    'fu': Vector3(),
                    'fz': Vector3()}

        self.fru = {'near':Vector3(),
                     'far': Vector3(),
                     'left': Vector3(),
                     'right': Vector3(),
                     'top': Vector3(),
                     'bottom': Vector3(),}

        self.pl_dist_far = float(far_depth_dist)
        self.pl_dist_near = float(near_depth_dist)

        self.fc_w = vwidth
        self.fc_h = vheight
        self.nc_w = vwidth/2
        self.nc_h = vheight/2

        self.origin = Vector3()
        self.clipped_state = 0

    def setFarPlane(self, origin, mat):
        """origin is Vector3
        mat is camera_static
        six hours to get this (or several)

        (-0.847213, 0, -41.964047)
        ('near', Vector3(-6.66133814775e-16, 0.0, -32.0))
        ('far', Vector3(-1.69442644284, 0.0, -51.9280937129))
        """

        self.origin = camera_matrix.get_row_vec3(3) #origin.copy()
        ex = camera_matrix.get_row_vec3(0).normalize()
        ey = camera_matrix.get_row_vec3(1).normalize()
        ez = (ex.cross(ey)*-1).normalize() #(origin-camera_matrix.get_row_vec3(3)).normalize() #so many issues here

        self.up = ey
        self.right = ex
        self.d = ez

        self.fpl['fr'] = ex
        self.fpl['fu'] = ey
        self.fpl['fz'] = ez

        self.fpl['nc'] = self.origin + ez * self.pl_dist_near #this is a point (these)
        self.fpl['fc'] = self.origin + ez * self.pl_dist_far

        # Normals:
        nc = self.fpl['nc'].copy()
        fc = self.fpl['fc'].copy()
        #print ez
        #aux = pc - (pc + (ex * 10.0))
        #auy = self.fpl['nc'] - (self.fpl['nc'] + (ey * 10.0))

        self.fru['near_d'] = ez.dot(nc)#*-1.0
        self.fru['near'] = ez #nc.normalize() #ez * self.origin

        self.fru['far_d'] = ez.dot(fc)
        self.fru['far'] = ez #fc.normalize() #-ez #(nc - fc).normalize() #* self.fpl['fc'] #.normalize() #-ez * self.origin


        # a = (self.fpl['nc'] + self.right * self.nc_w / 2) - self.origin
        # a.normalize()
        # self.fru['right'] = self.up * a
        """
        	public function setNormalAndPoint( normal:Number3D, pt:Number3D ):void
		{
			this.normal = normal;
			this.d = -Number3D.dot(normal, pt);
		}

		/**
		 * distance of point to plane.
		 *
		 * @param	v
		 * @return
		 */
		public function distance( pt:* ):Number
		{
			var p:Number3D = pt is Vertex3D ? pt.toNumber3D() : pt;
			return Number3D.dot(p, normal) + d;
		}


		"""

        # self.fpl['lt'] = self.fpl['fc'] + (ey * self.fc_h / 2) - (ex * self.fc_w / 2)
        # self.fpl['rt'] = self.fpl['fc'] + (ey * self.fc_h / 2) + (ex * self.fc_w / 2)
        # self.fpl['lb'] = self.fpl['fc'] - (ey * self.fc_h / 2) - (ex * self.fc_w / 2)
        # self.fpl['rb'] = self.fpl['fc'] - (ey * self.fc_h / 2) + (ex * self.fc_w / 2)
        #
        #
        # vx = self.fpl['rt'] - self.fpl['lt']
        # vy = self.fpl['lb'] - self.fpl['lt']

        #self.fru['far'] = vx.cross(vy) #.normalize()


    def showBounds(self):
        u_gl_list = glGenLists(1)
        glNewList(u_gl_list, GL_COMPILE)

        glPushMatrix()
        # o = self.fpl['fc']
        # glColor3f(0.50, 0.50, 0.50)
        # glTranslate(o.x, o.y, o.z)
        # glutWireCube(4.0)
        # glPopMatrix()
        o = self.origin *-1
        glTranslate(o.x, o.y, o.z)

        #glPushMatrix()
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 1.0)
        for pte in self.fpl:
            pt = self.fpl[pte]
            glVertex3f(pt.x, pt.y, pt.z)
        glEnd()

        glColor3f(0.0, 1.0, 0.0)

        glBegin(GL_LINES)
        pt = self.fpl['lt']
        glVertex3f(pt.x, pt.y, pt.z)
        pt = self.fpl['rb']
        glVertex3f(pt.x, pt.y, pt.z)
        glEnd()

        glBegin(GL_LINES)
        pt = self.fpl['rt']
        glVertex3f(pt.x, pt.y, pt.z)
        pt = self.fpl['lb']
        glVertex3f(pt.x, pt.y, pt.z)
        glEnd()

        #glPopMatrix()
        glPopMatrix()

        glEndList()
        return u_gl_list
        pass


    def testBounds(self, point):
        """dist = A*rx + B*ry + C*rz + D = n . r  + D"""
        # point translate to local space (relativize point)
        eset = ['far','near']
        self.clipped_state = 1
        for p in eset:
            dist = self.fru[p].dot(-point)+self.fru[p+'_d']
            if dist < 0: #zero is red
                self.clipped_state = 0
                return 0
        return self.clipped_state


box = viewClipBox(10.0, 20.0, float(width)/64, float(height)/64)
print(float(width), float(height))


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

while 1:
    time_passed = clock.tick()
    time_passed_seconds = time_passed / 1000.
    pressed = pygame.key.get_pressed()
    mouseevents = 'mouseevt'
    ect = 0
    
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: zpos = max(0.1, zpos-0.1)
            elif e.button == 5: zpos += 0.1
            elif e.button == 1: rotate = True
            elif e.button == 3: move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j
        ect += 1
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    spiel = "ot "
    odel = 0

    for k in pressed:
        if k:
            presscounter += 0.01
            spiel += pygame.key.name(odel)+","
        odel += 1
    
    presscounter *= 0.99
    #///print presscounter
    # Reset rotation and movement directions

    rotation_direction.set(0.0, 0.0, 0.0)
    movement_direction.set(0.0, 0.0, 0.0)

    # Modify direction vectors for key presses
    if pressed[K_LEFT]:
        rotation_direction.y = +rs
    elif pressed[K_RIGHT]:
        rotation_direction.y = -rs
    if pressed[K_UP]:
        rotation_direction.x = +rs
    elif pressed[K_DOWN]:
        rotation_direction.x = -rs
    if pressed[K_q]:
        rotation_direction.z = -rs
    elif pressed[K_e]:
        rotation_direction.z = +rs            
    if pressed[K_w]:
        movement_direction.z = -ms
    elif pressed[K_s]:
        movement_direction.z = +ms
    if pressed[K_a]:
        movement_direction.x = -ms
    elif pressed[K_d]:
        movement_direction.x = +ms
    

    #rotation_direction *= 0.98
    #movement_direction *= 0.998



    # alpha += rotation_direction.x * rotation_speed * time_passed_seconds
    # beta += rotation_direction.y * rotation_speed * time_passed_seconds
    # gamma += rotation_direction.z * rotation_speed * time_passed_seconds
    # * rotation_speed
    #movement_position__iadd__(movement_direction)

    user_pos_vector = user_matrix.get_row_vec3(3)

    box.setFarPlane(user_pos_vector, camera_static)

    box.testBounds(canonical_surface_marker)




    if odel is not 0:
        # Calculate rotation matrix and multiply by camera matrix
        rotation = rotation_direction * rotation_speed * time_passed_seconds
        rotation_matrix = Matrix44.xyz_rotation(*rotation)
        camera_matrix *= rotation_matrix
        camera_static = camera_matrix.copy()
        camera_static.get_inverse_rot_trans()

        direc = Vector3(camera_matrix.forward)
        u = user_pos_vector + (direc * camera_distance)
        camera_matrix.set_row(3,[u.x, u.y, u.z])



    #orientation = Quaternion.from_matrix(camera_matrix)

    #camera_q = Quaternion().camera_matrix

    #ass_matrix *= rotation_matrix

    #ass_matrix.invert()

    # user_matrix *= rotation_matrix
    #ship_matrix *= rotation_matrix
    #have the untranslated rotation_matrix
    #ship_rotation = rotation_direction * rotation_speed * time_passed_seconds
    #ship_rotation_matrix = Matrix44.xyz_rotation(*ship_rotation)
    #ship_matrix *= ship_rotation_matrix

    base = ship_matrix.get_row_vec3(3) #Vector3(ship_matrix.translate)

    # Calcluate movment and add it to camera matrix translate for z
    heading = Vector3(user_matrix.forward)
    movement = heading * movement_direction.z * movement_speed
    ship_matrix.translate += movement * time_passed_seconds

    # Calcluate movment and add it to camera matrix translate for z
    heading_r = Vector3(user_matrix.right)
    movement = heading_r * movement_direction.x * movement_speed
    ship_matrix.translate += movement * time_passed_seconds

    delta = ship_matrix.get_row_vec3(3) - base
    inertial_direction += delta * 0.01
    inertial_direction *= 0.99

    current_sector, height_in_sector, sector_normal = get_vertex_id(user_pos_vector, 0.1, current_sector)
    surface_position = height_in_sector * (user_pos_vector)
    dist_from_c = user_pos_vector.get_length()
    dist_from_surface = dist_from_c-surface_position.get_length()



    # gravity_constant_number = 0.0005
    # inertial_direction += upv.normalize()*gravity_constant_number
    #
    #
    # if dist_from_surface < 0.5:
    #     #bounce here!
    #     N = sector_normal.copy().normalise()
    #     I = inertial_direction.copy()
    #     R = Vector3()
    #     B = 2*(-I.dot(N))*N + I
    #     inertial_direction = B
    #


    user_matrix.translate += -inertial_direction






    dret = paint_a_gon(obj, current_sector)
    glCallList(dret)

    glLightfv(GL_LIGHT0, GL_POSITION,  (0, 200, 0, 1.0))
    # glLightfv(GL_LIGHT0+1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))



    #going to have to compare here.
    #glCallList(obj.gl_list)





    if trace_counter < trace_counter_max-1:
        trace_counter += 1
    else:
        TRACE.vertices.append(user_matrix.translate)
        if len(TRACE.vertices) > trace_length_max: TRACE.vertices.pop(0)
        trace_counter = 0



    glCallList(draw_line(user_matrix, surface_position, 2.0))
    glCallList(TRACE.draw(user_matrix))
    glCallList(draw_user(user_matrix))
    #glCallList(draw_ship(inertial_direction))
    glCallList(draw_anchor(user_matrix))


    glCallList(draw_camera_view(None, None))
    glCallList(box.showBounds())

    #display some text :)
    #glTranslate(-0.8,0.0,0.0)
    glPushMatrix()
    glTranslate(0.8, 0.0, 0.0)
    glScalef(0.002,0.002,0.002)
    debug('DC:'+str(dist_from_c)[0:4]+' DS:'+str(dist_from_surface)[0:4]+' SC:'+str(current_sector))
    debug(str(user_pos_vector))
    glPopMatrix()





    # camera_matrix = user_matrix.copy()







      #  direc * camera_distance

    # glTranslate(0.0, 0.0, 0.0)


    # glTranslate(user_pos_vector.x,user_pos_vector.y,user_pos_vector.z)

    # glCallList(draw_camera_view(camera_matrix))

    #glTranslate(0.0, 0.0, 1.0)

    # camera_static.translate





    glLoadMatrixd(camera_matrix.get_inverse().to_opengl())





    if polytimer < polytimer_max-1:
        polytimer += 1
    else:
        polytimer = 0

    if root_timer > 1000:
        root_timer = 0
    root_timer += 1

    pygame.display.flip()
    
    
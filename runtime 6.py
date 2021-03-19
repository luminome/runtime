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




glEnable(GL_LIGHTING)

glEnable(GL_LIGHT0)

glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))


glEnable(GL_LIGHT1)

glLightfv(GL_LIGHT1, GL_POSITION,  (0, 0, 0, 1.0))
glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 1.0, 0.5, 0.7))
glLightfv(GL_LIGHT1, GL_SPECULAR, (0.8, 1.0, 0.8, 1.0))


glEnable(GL_POINT_SMOOTH)
glEnable(GL_PROGRAM_POINT_SIZE)

glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)


glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded
 
# LOAD OBJECT AFTER PYGAME INIT

obj = OBJ('untitled.obj', swapyz=True)

subobj = OBJ('untitled-sub.obj', swapyz=True)




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





model_world_radius = 32.0
camera_distance = 10.0

# Camera transform matrix
camera_matrix = Matrix44()
#camera_matrix.translate = (0, 0, model_world_radius-camera_distance)
camera_static = Matrix44()


# User transform matrix
user_matrix = Matrix44()
user_matrix.translate = (0, 0, model_world_radius+camera_distance)

# User transform matrix
ship_matrix = Matrix44()
ship_matrix.translate = (0.0, 0.0, model_world_radius)

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


current_vert_map = []
canonical_surface_marker = Vector3()
lockvector = Vector3()

##################################################################################################
##################################################################################################
####################################### U T I L I T Y ############################################
##################################################################################################
##################################################################################################

class Trace(object):
    """visual trace (dashed line)"""
    trace_counter = 0
    trace_frame_interval = 5
    trace_length_max = 200


    def draw(self, env_mat, inlength):
        """draws visible tail/trace of positions"""
        if self.trace_counter == self.trace_frame_interval:
            if len(self.vertices) >= trace_length_max: self.vertices = self.vertices[2:]
            self.vertices.append(env_mat.get_row(3))
            self.trace_counter = 0
        else:
            if inlength > 0.001:
                self.trace_counter += 1

        t_list = glGenLists(2)
        glNewList(t_list, GL_COMPILE)
        glBegin(GL_LINES)
        glColor3f(0.1, 0.8, 0.1)
        oxy = 2
        for zv in self.vertices:
            if oxy < len(self.vertices):
                v = self.vertices[oxy]
                p = self.vertices[oxy-1]
                glVertex3f(v[0], v[1], v[2])
                glVertex3f(p[0], p[1], p[2])
                oxy += 2
        glEnd()
        glEndList()


        return t_list
    
    def __init__(self):
        self.vertices = []
        self.append = [(0.0, 0.0, 0.0)]

TRACE = Trace()

class SectorHandle(object):
    """the lookup function/class"""

    def get_polygon_intersect(self):
        """check poly against existing address:
        returns [get_vertex_h, get_poly_norm, poly_vertices, sector_string, poly_number, error]"""

        vert_set = []
        poly_verts = []
        scope_normalized = 0.1

        for kk in self.selected_vert_map:
            v = POLY_CENTERS_ARRAY[kk]['center'].copy().normalize()
            d = self.seek_vector_normal.get_distance_to(v)
            if d < scope_normalized:
                vert_set.append(kk)
        if len(vert_set) == 0: return 0

        for v in vert_set:
            traf = Vector3()
            dve = obj.faces[v][0]
            va = Vector3(obj.vertices[dve[0] - 1])
            vb = Vector3(obj.vertices[dve[1] - 1])
            vc = Vector3(obj.vertices[dve[2] - 1])
            d = intersect_test(traf, self.seek_vector, va, vb, vc)
            poly_verts = [va, vb, vc]
            if d:
                self.sub_sector_poly_number = v
                return [d[0],d[1],poly_verts,self.sector_str,int(v),self.w_error]

        if len(poly_verts) == 0: return 0
        pass

    def get_address_from_dict(self, address_dict):
        """validate/check for vector in address dictionaries"""
        if 'sectors' in address_dict[0]:
            aps = []
            for kk in address_dict:
                #pretty vague
                vb = address_dict[kk]['center']
                d = vb.get_distance_to(self.seek_vector)
                aps.append((d, kk))
            d = min(aps)
            select_a = d[1]
            select_id = address_dict[d[1]]['id']
            return select_a, select_id
        else:
            for kk in address_dict:
                #very specific
                traf = Vector3()
                dve = address_dict[kk]['v']
                d = intersect_test(traf, self.seek_vector, dve[0], dve[1], dve[2])
                if d and (d[0] > 0.0):
                    select_a = kk
                    select_id = address_dict[kk]['id']
                    return select_a, select_id
        pass
    
    def get_address(self):
        """lookup address (recurs)"""
        sel, sector_id = self.get_address_from_dict(SUBSECTOR_ARRAY)
        sel_sub, sub_sector_id = self.get_address_from_dict(SUBSECTOR_ARRAY[sel]['sectors'])
        self.selected_vert_map = SUBSECTOR_ARRAY[sel]['sectors'][sel_sub]['vertmap']
        self.sector_number = sel
        self.sub_sector_number = sel_sub
        self.sector_str = [sector_id,sub_sector_id]
        pass

    def locate(self, vector):
        self.seek_vector = vector.copy()
        self.seek_vector_normal = vector.copy().normalize()
        self.w_error = 0
        intersect = self.get_polygon_intersect()

        if intersect:
            return intersect
        else:
            self.get_address()
            self.w_error = "new sector address "+str(self.sector_str)
            return self.get_polygon_intersect()

        pass

    def __init__(self):
        print('created Sector_Handle(r)')
        self.w_error = None
        self.sector_number = 0
        self.sub_sector_number = 0
        self.selected_vert_map = []
        self.seek_vector = Vector3()
        self.seek_vector_normal = Vector3()

SHT = SectorHandle()



def get_closest_address_or_vector(v, group):
    #get the closest one, not the "near" ones
    #needed to intersect test bc distances too vague
    if 'sectors' in group[0]:
        aps = []
        for kk in group:
            #pretty vague
            vb = group[kk]['center']
            d = vb.get_distance_to(v)
            aps.append((d, kk))
        d = min(aps)
        select_a = d[1]
        select_id = group[d[1]]['id']
        return select_a, select_id
    else:
        for kk in group:
            #very specific
            traf = Vector3()
            dve = group[kk]['v']
            d = intersect_test(traf, v, dve[0], dve[1], dve[2])
            if d and (d[0]>0.0):
                select_a = kk
                select_id = group[kk]['id']
                return select_a, select_id



def get_vertex_id(at_vector, scope_normalized, c_sector):
    vert_set = []
    cs = 0#c_sector
    get_poly = []
    get_vertex_h = 0
    get_poly_norm = Vector3()
    avc = at_vector.copy().normalize()
    w_error = 0


    sel_sector, sector_id = get_closest_address_or_vector(at_vector, SUBSECTOR_ARRAY)

    sel_sub_sector, sub_sector_id = get_closest_address_or_vector(at_vector, SUBSECTOR_ARRAY[sel_sector]['sectors'])

    vert_map = SUBSECTOR_ARRAY[sel_sector]['sectors'][sel_sub_sector]['vertmap']

    if c_sector not in vert_map:
        w_error = str(c_sector)+'c_sector not in'

    nsectors = [sel_sector, sel_sub_sector, sector_id, sub_sector_id]

    for kk in vert_map:
        v = POLY_CENTERS_ARRAY[kk]['center'].copy().normalize()
        d = avc.get_distance_to(v)
        if d < scope_normalized:
            vert_set.append(kk)

    for v in vert_set:
        traf = Vector3()
        dve = obj.faces[v][0]
        va = Vector3(obj.vertices[dve[0] - 1])
        vb = Vector3(obj.vertices[dve[1] - 1])
        vc = Vector3(obj.vertices[dve[2] - 1])
        d = intersect_test(traf, at_vector, va, vb, vc)
        get_poly = [va, vb, vc]

        if d:
            get_poly_norm = d[1]
            get_vertex_h = d[0]
            cs = int(v)
            break


    nsectors.append(cs)
    if get_vertex_h == 0: w_error = [nsectors]

    return [get_poly, get_vertex_h, get_poly_norm, nsectors, w_error]


def intersect_test(p, d, V0, V1, V2):
    """http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/"""

    e1 = V1.__sub__(V0)
    e2 = V2.__sub__(V0)
    h = d.cross(e2)
    a = e1.dot(h)

    if (a > -0.00001) and (a < 0.00001): return False

    f = 1 / a
    s = p.__sub__(V0)
    u = f * s.dot(h)

    if (u < 0.0) or (u > 1.0): return False

    q = s.cross(e1)
    v = f * d.dot(q)

    if (v < 0.0) or (u + v > 1.0): return False

    t = f * e2.dot(q)
    n = e1.cross(e2)

    if abs(t) > 0.00001:
        return (t, n)
    else:
        return False


##################################################################################################
##################################################################################################
############################## W O R L D - M O D E L - D R A W ###################################
##################################################################################################
##################################################################################################


POLY_CENTERS_ARRAY = {}
SUBSECTOR_ARRAY = {}


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
    #now we have two specific models.
    """have vertices for siplified pointset...
    must put higher=res points under that."""

    """construct center-point vertices for all faces.
    populate POLY_CENTERS_ARRAY dictionary in form: k:main_name_int, v:vector
    prints some metrics."""

    index = 0

    vct = 0
    redux = 0
    vertex_trace = {}

    for face in gon_obj.faces:
        vertices, normals, texture_coords, material = face
        #print(face)
        POLY_CENTERS_ARRAY[index] = {}
        v1 = Vector3()
        for ii in range(len(vertices)):
            vg = Vector3((gon_obj.vertices[vertices[ii] - 1]))
            v1 += vg
        v1 /= 3.0
        POLY_CENTERS_ARRAY[index]['center'] = v1
        index += 1

    #print vertex_trace
    #print(POLY_REDUX_ONE)
    fl = len(POLY_CENTERS_ARRAY)
    print("POLY_CENTERS_ARRAY", fl, POLY_CENTERS_ARRAY[fl - 1])
    pass

# build_polygon_topography(obj, 'nodal_random', 12)
#build_polygon_topography(obj, 'arbitrary_random', 0.01)
#obj.refresh()

build_polygon_centers(obj)


def build_sector_addresses(thesubobj):
    """iterate over faces in submodel icosahedron
    split each face into four sub-polys
    run intersection tests on face and sub-polys
    save this information for addressing
    requires that POLY_CENTERS_ARRAY be set"""
    glColor4f(0.8, 0.8, 0.8, 0.1)
    index = 0
    def zip_face(fv):
        return ((fv[0] / 3.0 + fv[1] / 3.0 + fv[2] / 3.0)) #.normalize() * model_world_radius

    for face in thesubobj.faces:
        SUBSECTOR_ARRAY[index] = {}
        dve = face[0]
        va = Vector3(thesubobj.vertices[dve[0] - 1])
        vb = Vector3(thesubobj.vertices[dve[1] - 1])
        vc = Vector3(thesubobj.vertices[dve[2] - 1])

        vam = ((va / 2.0 + vb / 2.0))
        vbm = ((vb / 2.0 + vc / 2.0))
        vcm = ((vc / 2.0 + va / 2.0))

        SUBSECTOR_ARRAY[index] = {'id': str(index)+'iso','center': zip_face([va, vb, vc]), 'sectors': {}, 'v':[va, vb, vc]}
        SUBSECTOR_ARRAY[index]['sectors'] = {0:{'id': str(index)+'a', 'center': zip_face([va, vam, vcm]), 'vertmap': [], 'v': [va, vam, vcm]},
                                             1:{'id': str(index)+'b', 'center': zip_face([vam, vb, vbm]), 'vertmap': [], 'v': [vam, vb, vbm]},
                                             2:{'id': str(index)+'c', 'center': zip_face([vbm, vc, vcm]), 'vertmap': [], 'v': [vbm, vc, vcm]},
                                             3:{'id': str(index)+'d', 'center': zip_face([vam, vbm, vcm]), 'vertmap': [], 'v': [vam, vbm, vcm]}}

        for k in SUBSECTOR_ARRAY[index]['sectors']:
            ksub = SUBSECTOR_ARRAY[index]['sectors'][k]
            va = ksub['v'][0]
            vb = ksub['v'][1]
            vc = ksub['v'][2]

            for kk in POLY_CENTERS_ARRAY:
                nil = Vector3(0.0,0.0,0.0)
                v = POLY_CENTERS_ARRAY[kk]['center']
                d = intersect_test(nil, v, va, vb, vc)
                if d and (d[0] > 0.0):
                    ksub['vertmap'].append(kk)

        print(SUBSECTOR_ARRAY[index]['id'])
        index += 1


build_sector_addresses(subobj)

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

def draw_subsector_poly_centers():
    #POLY_CENTERS_ARRAY
    gl_pc_list = glGenLists(2)
    glNewList(gl_pc_list, GL_COMPILE)
    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)

    for kk in current_vert_map:
        v = POLY_CENTERS_ARRAY[kk]['center'].copy()*1.001
        glVertex3f(v.x, v.y, v.z)

    glEnd()
    glEndList()
    return gl_pc_list

def draw_sub_verts():
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    glBegin(GL_LINES)

    for blob in SUBSECTOR_ARRAY:
        glColor3f(1.0, 1.0, 1.0)
        # vg = Vector3(SUBSECTOR_ARRAY[blob]['center'])
        # glVertex3f(vg[0], vg[1], vg[2])
        # vg *= 1.3
        # #glVertex3f(0.0,0.0,0.0)
        # glVertex3f(vg[0], vg[1], vg[2])

        for sb in SUBSECTOR_ARRAY[blob]['sectors']:
            glColor3f(1.0, 0.0, 1.0)
            sub_blob = SUBSECTOR_ARRAY[blob]['sectors'][sb]
            vg = Vector3(sub_blob['center'])
            glVertex3f(vg[0], vg[1], vg[2])
            vg *= 1.2
            #glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(vg[0], vg[1], vg[2])

            a = (sub_blob['v'][0]*1.01).as_tuple()
            b = (sub_blob['v'][1]*1.01).as_tuple()
            c = (sub_blob['v'][2]*1.01).as_tuple()

            glVertex3fv(a)
            glVertex3fv(b)

            glVertex3fv(b)
            glVertex3fv(c)

            glVertex3fv(c)
            glVertex3fv(a)




    glEnd()
    glEndList()
    return gl_sub_list

def draw_sector_poly_hilight(gon_obj, index):
    """used to indicate the current sector"""
    gl_list = glGenLists(2)
    glNewList(gl_list, GL_COMPILE)
    #glFrontFace(GL_CW)
    
    vertices, normals, texture_coords, material = gon_obj.faces[index]
    glColor3f(0.2, 1.0, 0.2)
    a = gon_obj.vertices[vertices[0] - 1]
    b = gon_obj.vertices[vertices[1] - 1]
    c = gon_obj.vertices[vertices[2] - 1]

    glBegin(GL_LINES)

    glVertex3fv(a)
    glVertex3fv(b)

    glVertex3fv(b)
    glVertex3fv(c)

    glVertex3fv(c)
    glVertex3fv(a)

    # for ii in range(len(vertices)):
    #     glVertex3fv(gon_obj.vertices[vertices[ii] - 1])
    #
    glEnd()
    glEndList()
        
    return gl_list

def draw_user(atmat):
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)

    glColor4f(0.0, 1.0, 0.1, 0.4)

    t = atmat.translate
    v = Vector3(t[0],t[1],t[2])

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

    #
    # dl = 32
    # o = 0.2
    #
    # bx = abs(atmat.x*dl)/2
    # by = abs(atmat.y*dl)/2
    # bz = abs(atmat.z*dl)/2
    #
    # if atmat.x<0:bx *= -1
    # posx = Vector3((o*bx), 0.0, 0.0)
    # draw_pologon_marker(posx, abs(atmat.x*dl))
    #
    # if atmat.y<0:by *= -1
    # posy = Vector3(0.0, (o*by), 0.0)
    # draw_pologon_marker(posy, abs(atmat.y*dl))
    #
    # if atmat.z<0:bz *= -1
    # posz = Vector3(0.0, 0.0,(o*bz))
    # draw_pologon_marker(posz, abs(atmat.z*dl))

    glEndList()
    return s_gl_list

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

def debug(lines):
    size = 0.25
    sca = (1 / 152.2) * size
    glPushMatrix()
    glColor4f(0.0, 1.0, 0.0, 0.4)
    glLoadIdentity()
    glRotate(180,0.0,1.0,0.0)
    glTranslate(1.0, len(lines)*size, 9.0)
    linect = 0
    for chars in lines:
        glPushMatrix()
        glTranslate(0.0, linect * -size, 0.0)
        glScalef(sca, sca, sca)
        for v in str(chars):
            glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN, ord(v))
        glPopMatrix()
        linect += 1
    glPopMatrix()

def draw_line(atmat, dpos):
    v = atmat.get_row_vec3(3)

    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)

    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(dpos.x, dpos.y, dpos.z)
    glVertex3f(0.0, 0.0, 0.0)
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0.2, 0.8, 0.2)
    glVertex3f(0.0, 0.0, 0.0)
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

def draw_utility_line(origin,a,b):
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)
    glPushMatrix()
    glTranslate(origin.x, origin.y, origin.z)
    glColor3f(1.0, 1.0, 1.0)

    glBegin(GL_LINES)
    glVertex3f(0.0,0.0,0.0)
    glVertex3f(a.x,a.y,a.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x,b.y,b.z)
    glEnd()
    glPopMatrix()
    glEndList()

    return a_gl_list



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
        self.origin = camera_matrix.get_row_vec3(3) #origin.copy()
        ex = camera_matrix.get_row_vec3(0).normalize()
        ey = camera_matrix.get_row_vec3(1).normalize()
        ez = (ex.cross(ey)*-1).normalize()
        self.fpl['nc'] = self.origin + ez * self.pl_dist_near #this is a point (these)
        self.fpl['fc'] = self.origin + ez * self.pl_dist_far

        self.fru['near_d'] = ez.dot(self.fpl['nc'])
        self.fru['near'] = ez
        self.fru['far_d'] = ez.dot(self.fpl['fc'])
        self.fru['far'] = ez


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
        eset = ['far','near']
        #self.clipped_state = 1

        for p in eset:
            dist = self.fru[p].dot(-point)+self.fru[p+'_d']
            if dist < 0: #zero is red
                #self.clipped_state = 0
                return 0

        return 1 #self.clipped_state

box = viewClipBox(20.0, 30.0, float(width)/64, float(height)/64)
print(float(width), float(height))

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

glMatrixMode(GL_MODELVIEW)
user_surface_bind = 0
sectors = []

while 1:
    time_passed = clock.tick()
    time_passed_seconds = time_passed / 1000.
    pressed = pygame.key.get_pressed()
    #mouseevents = 'mouseevt'
    ect = 0
    
    for e in pygame.event.get():
        #print(str(e.type), e)
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                sys.exit()
        elif e.type == 5:
            if e.button == 4: zpos = max(0.1, zpos-0.1)
            elif e.button == 5: zpos += 0.1
            elif e.button == 1: rotate = True
            elif e.button == 3: move = True
        elif e.type == 6:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == 4:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j
        ect += 1
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # spiel = "ot "
    #
    #
    # for k in pressed:
    #     if k:
    #         presscounter += 0.01
    #         spiel += pygame.key.name(odel)+","
    #     odel += 1
    #
    # presscounter *= 0.99
    #///print presscounter
    # Reset rotation and movement directions
    keydown = 0
    #print(pressed)
    if 1 in pressed: keydown = 1

    rotation_direction.set(0.0, 0.0, 0.0)
    movement_direction.set(0.0, 0.0, 0.0)

    # Modify direction vectors for key presses
    if pressed[K_LEFT]:
        rotation_direction.y = -rs
    elif pressed[K_RIGHT]:
        rotation_direction.y = +rs
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

    #box.setFarPlane(user_pos_vector, camera_static)

    #box.testBounds(canonical_surface_marker)




    if keydown != 0:
        # Calculate rotation matrix and multiply by camera matrix
        rotation = rotation_direction * rotation_speed * time_passed_seconds
        rotation_matrix = Matrix44.xyz_rotation(*rotation)
        camera_matrix *= rotation_matrix
        camera_static = camera_matrix.copy()
        camera_static.get_inverse_rot_trans()
        user_matrix *= rotation_matrix







        #ass_matrix.set_row(3, [0.0,0.0,0.0])
        #ass_matrix.translate = (0.0,0.0,0.0)

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
    heading = Vector3(user_matrix.forward)*-1
    movement = heading * movement_direction.z * movement_speed
    ship_matrix.translate += movement * time_passed_seconds
    # Calcluate movment and add it to camera matrix translate for z
    heading_r = Vector3(user_matrix.right)
    movement = heading_r * movement_direction.x * movement_speed
    ship_matrix.translate += movement * time_passed_seconds

    delta = ship_matrix.get_row_vec3(3) - base
    inertial_direction += delta * 0.015

    uvp = user_pos_vector.copy().normalize()


    # [get_vertex_h, get_poly_norm, poly_vertices, sector_string, poly_number, error]
    #sector_poly, height_in_sector, sector_normal, sectors, err = get_vertex_id(user_pos_vector, 0.1, current_sector)

    height_in_sector, sector_normal, sector_poly, sectors, current_sector, sh_err = SHT.locate(user_pos_vector)

    if sh_err:
        print('error',sh_err)


    surface_position = height_in_sector * (user_pos_vector)
    # canonical_surface_marker = surface_position.copy()

    dist_from_c = user_pos_vector.get_length()
    dist_from_surface = dist_from_c-surface_position.get_length()

    current_vert_map = SHT.selected_vert_map #SUBSECTOR_ARRAY[sectors[0]]['sectors'][sectors[1]]['vertmap']

    uvp = user_pos_vector.copy().normalize()

    #gravity_constant_number = 0.001
    inertial_direction *= 0.99
    #inertial_direction += uvp * gravity_constant_number

    inert_dir = inertial_direction.get_length()


    N = sector_normal.copy().normalize()

    #PN = N.cross(sector_poly[0]).normalize()

    uvi = inertial_direction.copy().normalize()

    A = acos(N.dot(uvi))-((pi/180)*90)
    #this is perpendicular to te offset between plane normal and inertial direction

    glCallList(draw_utility_line(surface_position, N, -N))
    #glCallList(draw_utility_line(surface_position, uvi*-2, uvi*2))

    # inert_dir = inertial_direction.length


    if user_surface_bind:
        if surface_position.length:
            ed = (surface_position.length + 0.25) * uvp
            user_matrix.translate = ed
    else:
        if dist_from_surface < 0.2:
            #degrees(A) > 45.0:
            if inert_dir > 0.01:
                #bounce here!
                I = inertial_direction.copy()
                B = 2*(-I.dot(N))*N + I
                inertial_direction = B

            else:
                user_surface_bind = 1



    user_matrix.translate += -inertial_direction
    inertial_direction *= 0.99

    light_d = user_pos_vector * 1.1
    glLightfv(GL_LIGHT0, GL_POSITION,  (0, 200, 0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))


    glCallList(draw_sector_poly_hilight(obj, current_sector))
    glCallList(TRACE.draw(user_matrix, inert_dir))


    glCallList(obj.gl_list)
    #glCallList(subobj.gl_list)
    #glCallList(draw_sub_verts())
    #glCallList(box.showBounds())

    glCallList(draw_subsector_poly_centers())
    glCallList(draw_line(user_matrix, surface_position))
    glCallList(draw_user(user_matrix))
    glCallList(draw_ship(inertial_direction))

    debug(['DC:' + str(dist_from_c)[0:4],
           'DS:' + str(dist_from_surface)[0:4],
           'sector:' + str(sectors)+' poly:'+str(current_sector),
           'inert:' + str(inert_dir)[0:6],
           'angle:' + str(degrees(A))[0:6],
           'bind:' + str(user_surface_bind)])


    #user_pos_vector = user_matrix.get_row_vec3(3)

    direc = Vector3(camera_matrix.forward)
    u = user_matrix.get_row_vec3(3) + (direc * camera_distance)
    camera_matrix.set_row(3,[u.x, u.y, u.z])

    glLoadMatrixd(camera_matrix.get_inverse().to_opengl())

    pygame.display.flip()
    
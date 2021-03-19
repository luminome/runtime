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
clock = pygame.time.Clock()
viewport = (1200,800)
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


glShadeModel(GL_SMOOTH)# most obj files expect to be smooth-shaded
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

### P R O J E C T I O N ###
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glFrustum (-1, 1, -1, 1, 3.0, 10.0)
gluLookAt (0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
width, height = viewport
gluPerspective(75, width/float(height), 1, 100.0)
### P R O J E C T I O N ###


obj = OBJ('untitled.obj', swapyz=True)
subobj = OBJ('untitled-sub.obj', swapyz=True)

print(str(len(obj.faces)) + ' faces')
print(str(len(obj.normals)) + ' normals')
print(str(len(obj.vertices)) + ' vertices')


model_world_radius = 32.0
camera_distance = 6.0

camera_matrix = Matrix44()
user_surface_matrix = Matrix44()
user_matrix = Matrix44()
arse_mat = Matrix44()

user_matrix.translate = (0.00, 0, model_world_radius+camera_distance)
#user_matrix.set_row(1,[0.0,-1.0,0.0])

ship_direction_vector = Vector3(0.0,0.0,0.0)
inertial_direction = Vector3()

rotation_direction = Vector3()
rotation_speed = radians(90.0)

movement_position = Vector3()
movement_direction = Vector3()
movement_speed = 0.5
movement_speed_decay = 0.9


rs = 1 #rotation increment
ms = 1 #motion increment
rotate = move = False



##################################################################################################
##################################################################################################
####################################### U T I L I T Y ############################################
##################################################################################################
##################################################################################################


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


class Trace(object):
    """visual trace (dashed line)"""
    trace_counter = 0
    trace_frame_interval = 1
    trace_length_max = 400

    def draw(self, env_mat):
        """draws visible tail/trace of positions"""
        if self.trace_counter == self.trace_frame_interval:
            if len(self.vertices) >= self.trace_length_max: self.vertices = self.vertices[2:]
            self.vertices.append(env_mat.get_row(3))
            self.trace_counter = 0
        else:
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
            dve = obj.faces[v][0]
            va = Vector3(obj.vertices[dve[0] - 1])
            vb = Vector3(obj.vertices[dve[1] - 1])
            vc = Vector3(obj.vertices[dve[2] - 1])
            d = intersect_test(self.seek_origin, self.seek_vector, va, vb, vc)
            poly_verts = [va, vb, vc]
            if d:
                self.sub_sector_poly_number = v
                return [d[0],d[1],poly_verts,self.sector_str,int(v),self.w_error]

        if len(poly_verts) == 0: return 0
        pass

    def get_address_from_dict(self, address_dict, loctype):
        """validate/check for vector in address dictionaries"""
        if loctype == 'BY_DISTANCE':
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

        elif loctype == 'BY_INTERSECT':
            for kk in address_dict:
                #very specific
                dve = address_dict[kk]['v']
                d = intersect_test(self.seek_origin, self.seek_vector, dve[0], dve[1], dve[2])
                if d and (d[0] > 0.0):
                    select_a = kk
                    select_id = address_dict[kk]['id']
                    return select_a, select_id

        elif loctype == 'BY_CLIP_BOUNDS':
            BOX.setBoundsScale(0.75)
            in_view = []
            verts = []
            for kk in address_dict:
                #pretty vague
                vb = kk['p']
                if BOX.testBounds(vb):
                    in_view.append(kk['id'])
                    verts += SUBSECTOR_ARRAY[kk['ref'][0]]['sectors'][kk['ref'][1]]['vertmap']
            self.message = str((len(verts),'selected'))
            BOX.setBoundsScale(0.3)
            self.aux_vert_map = [x for x in verts if BOX.testBounds(POLY_CENTERS_ARRAY[x]['center'])]
            pass
        else:
            return 0, 'no address'

    def get_address(self):
        """lookup address (recurs)"""
        sel, sector_id = self.get_address_from_dict(SUBSECTOR_ARRAY,'BY_DISTANCE')
        sel_sub, sub_sector_id = self.get_address_from_dict(SUBSECTOR_ARRAY[sel]['sectors'],'BY_INTERSECT')
        self.selected_vert_map = SUBSECTOR_ARRAY[sel]['sectors'][sel_sub]['vertmap']
        self.sector_number = sel
        self.sub_sector_number = sel_sub
        self.sector_str = [sector_id,sub_sector_id]
        pass

    def locate(self, origin, vector):
        """SAUCE!"""
        self.seek_origin = origin.copy()
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

    def test_bounds(self):
        self.get_address_from_dict(SUBSECTOR_CENTERS_ARRAY, 'BY_CLIP_BOUNDS')



    def __init__(self):
        print('created Sector_Handle(r)')
        self.w_error = 'None'
        self.sector_number = 0
        self.sub_sector_number = 0
        self.selected_vert_map = []
        self.aux_vert_map = []
        self.seek_vector = Vector3()
        self.seek_vector_normal = Vector3()
        self.message = 'init'


class ViewClipBox(object):
    """Nasty frustrum culling subroutine"""
    def __init__(self, near_depth_dist, far_depth_dist, vwidth, vheight, init_scale):
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

        self.init_scale = init_scale
        self.pl_dist_far = float(far_depth_dist)
        self.pl_dist_near = float(near_depth_dist)

        self.init_width = vwidth
        self.init_height = vheight

        self.fc_w = (vwidth) * self.init_scale
        self.fc_h = (vheight) * self.init_scale
        self.nc_w = (vwidth/2) * self.init_scale
        self.nc_h = (vheight/2) * self.init_scale
        self.p = Vector3()

    def setBoundsScale(self, scale):
        """for the sake of addressing system"""
        if scale == 'default': scale = self.init_scale
        svwidth = self.init_width * scale
        svheight = self.init_height * scale
        self.fc_w = svwidth
        self.fc_h = svheight
        self.nc_w = svwidth/2
        self.nc_h = svheight/2
        self.setClipBounds(self.p, self.l, self.u)

    def setClipBounds(self, origin, look, up):
        #WAS//inverse rotational trans. makes sense that it works for n and f. but U is deciding factor in application
        #wc = Vector3(0.0,0.0,0.0)
        horizon_threshold = -16.0
        self.p = origin #camera_matrix.get_row_vec3(3) #origin.copy()
        self.l = look
        self.u = up

        Z = self.p - look
        Z.normalize()
        X = up.cross(Z)
        X.normalize()
        Y = X.cross(Z)
        Y.normalize()

        self.fpl['nc'] = self.p - Z * self.pl_dist_near
        self.fpl['fc'] = self.p - Z * self.pl_dist_far
        self.fru['near_d'] = Z.dot(self.fpl['nc'])
        self.fru['near'] = Z
        self.fru['far_d'] = -Z.dot(self.fpl['fc'])
        self.fru['far'] = -Z

        ht = Z * horizon_threshold
        self.fru['world_d'] = -Z.dot(ht)
        self.fru['world'] = -Z

        aux = self.fpl['nc'] - X * self.nc_w
        auxld = (aux - self.p).normalize()
        normal = auxld.cross(Y)
        self.fru['right_d'] = normal.dot(aux)
        self.fru['right'] = normal
        self.fpl['rtn'] = aux

        aux = self.fpl['nc'] + X * self.nc_w
        auxld = (aux - self.p).normalize()
        normal = auxld.cross(Y)
        self.fru['left_d'] = -normal.dot(aux)
        self.fru['left'] = -normal
        self.fpl['ltn'] = aux

        aux = self.fpl['nc'] - Y * self.nc_h
        auxld = (aux - self.p).normalize()
        normal = auxld.cross(X)
        self.fru['top_d'] = -normal.dot(aux)
        self.fru['top'] = -normal
        self.fpl['ttn'] = aux

        aux = self.fpl['nc'] + Y * self.nc_h
        auxld = (aux - self.p).normalize()
        normal = auxld.cross(X)
        self.fru['bottom_d'] = normal.dot(aux)
        self.fru['bottom'] = normal
        self.fpl['btn'] = aux


        ### REFERENCE ONLY ###
        # fc = self.fpl['fc'].copy()
        # nc = self.fpl['nc'].copy()
        #
        # self.fpl['ftl'] = fc + Y * self.fc_h - X * self.fc_w
        # self.fpl['ftr'] = fc + Y * self.fc_h + X * self.fc_w
        # self.fpl['fbl'] = fc - Y * self.fc_h - X * self.fc_w
        # self.fpl['fbr'] = fc - Y * self.fc_h + X * self.fc_w
        #
        # self.fpl['ntl'] = nc + Y * self.nc_h - X * self.nc_w
        # self.fpl['ntr'] = nc + Y * self.nc_h + X * self.nc_w
        # self.fpl['nbl'] = nc - Y * self.nc_h - X * self.nc_w
        # self.fpl['nbr'] = nc - Y * self.nc_h + X * self.nc_w

    def showBounds(self):
        u_gl_list = glGenLists(1)
        glNewList(u_gl_list, GL_COMPILE)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_POINTS)
        for pte in self.fpl:
            pt = self.fpl[pte]
            glVertex3f(pt.x, pt.y, pt.z)
        glEnd()
        glEndList()
        return u_gl_list
        pass

    def testBounds(self, point):
        eset = ['world','far','near','right','left','top','bottom']
        for plane in eset:
            dist = self.fru[plane].dot(point)-self.fru[plane+'_d']
            if dist > 0: return 0
        return 1


class RudePhysics(object):
    """Rudimentary Physics Handler.
    aggregate function for any pysical object in world"""

    gravity_c = -9.810/2.0

    def __init__(self):
        self.Accel = Vector3()
        self.Pos = Vector3()
        self.OldVelo = Vector3()
        self.Velo = Vector3()
        self.GravityVector = Vector3()
        self.Thrusters = Vector3()

    def update_pos(self, current_pos, delta_time):
        self.GravityVector = current_pos.normalize() * self.gravity_c
        self.Accel = self.GravityVector + self.Thrusters
        self.OldVelo = self.Velo.copy()
        self.Velo += self.Accel * delta_time
        self.Pos += (self.OldVelo + self.Velo) * 0.5 * delta_time
        return self.Pos

    def set(self, direction_vector):
        self.Thrusters = direction_vector

    def set_deccel(self):
        self.Velo *= 0.8
        pass

    def bounce(self, N):
        """N:surface intersected normal"""
        self.Velo *= 0.5
        I = self.Velo.copy()
        self.Velo =  2 * (-I.dot(N)) * N + I
        pass

    def creep(self, N):
        """N:surface intersected normal"""
        I = self.Velo.copy()
        self.Velo = (-I.dot(N)) * N + I
        pass


class Corey_Has_Smokes(object):
    def __init__(self):
        """smokes is an array"""
        self.smokes = []
        self.gl_list = []
        print("Corey_Has_Smokes.")

    def gimme_a_fucking_smoke(self, position, howmany):
        for l in range(0,howmany):
            a = random.random() - 0.5
            b = random.random() - 0.5
            c = random.random() - 0.5
            s = random.random()
            randy = Vector3(a,b,c)
            self.smokes.append([position, 1, randy, s])

    def show(self):
        gl_sub_list = glGenLists(2)
        glNewList(gl_sub_list, GL_COMPILE)
        glFrontFace(GL_CW)
        puff = 30
        for n in self.smokes:
            if n[1] > puff:
                self.smokes.pop(0)
            else:
                ds = sin((n[1] * pi) / puff)
                glPushMatrix()
                randy = ((n[2]*n[1])/10.0)
                glTranslate(n[0].x+randy.x, n[0].y+randy.y, n[0].z+randy.z)
                glColor4f(0.5, 0.5, 0.5, 1.0 - n[1]/puff)#1/n[1])
                glutSolidSphere((n[3]+ds)*0.2, 8, 8)
                glPopMatrix()
            n[1] += 1.0

        glEndList()
        return gl_sub_list






SMOKES = Corey_Has_Smokes()

TRACE = Trace()
SHT = SectorHandle()
SHZ = SectorHandle()

USER = RudePhysics()


#OH GOD IT WORKS NOW
N = camera_distance
F = camera_distance*6
BOX = ViewClipBox(N, F, float(width)/(F), float(height)/(F), 0.4)

##################################################################################################
##################################################################################################
############################## W O R L D - M O D E L - D R A W ###################################
##################################################################################################
##################################################################################################

STAR_ARRAY = []
POLY_CENTERS_ARRAY = {}
SUBSECTOR_ARRAY = {}
SUBSECTOR_CENTERS_ARRAY = []


def build_stars(howmany):
    m = 2.0
    w = 12000.0
    for n in range(0,howmany):
        #e = random.random()*m - m/2
        a = random.random()*m - m/2
        b = random.random()*m - m/2
        c = random.random()*m - m/2

        d = random.random()*w
        #print (a,b,c)
        starpos = Vector3(a,b,c)*d

        cc = model_world_radius * 2 * starpos.get_normalised()

        STAR_ARRAY.append(starpos+cc)

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
        vC = vB.cross(vA)
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

def build_sector_addresses(thesubobj):
    """iterate over faces in submodel icosahedron
    split each face into four sub-polys
    run intersection tests on face and sub-polys
    save this information for addressing
    requires that POLY_CENTERS_ARRAY be set"""
    glColor4f(0.8, 0.8, 0.8, 0.1)
    index = 0
    subindex = 0
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
            vertset = []
            for kk in POLY_CENTERS_ARRAY:
                nil = Vector3(0.0,0.0,0.0)
                v = POLY_CENTERS_ARRAY[kk]['center']
                d = intersect_test(nil, v, va, vb, vc)
                if d and (d[0] > 0.0):
                    ksub['vertmap'].append(kk)
                    dist = v.copy().normalize().get_distance_to(ksub['center'].copy().normalize())
                    vertset.append([dist,kk,v])

            ksub['z_center'] = min(vertset)[2]
            if k == 3: SUBSECTOR_ARRAY[index]['z_center'] = min(vertset)[2]

            sub_c = {'id':ksub['id'],
                     'p':ksub['z_center'],
                     'ref':(index,k)}

            SUBSECTOR_CENTERS_ARRAY.append(sub_c)
            #subindex += 1

        print(SUBSECTOR_ARRAY[index]['id'])
        index += 1


    for ele in SUBSECTOR_CENTERS_ARRAY:
        print(ele)



def make_subdivisions(start_element):
    pass





build_polygon_topography(obj, 'nodal_random', 16)
#build_polygon_topography(obj, 'arbitrary_random', 0.03)
obj.refresh()

build_polygon_centers(obj)

build_sector_addresses(subobj)

build_stars(1000)

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
    #selected_vert_map
    for kk in SHT.aux_vert_map:
        v = POLY_CENTERS_ARRAY[kk]['center'].copy()*1.001
        glVertex3f(v.x, v.y, v.z)

    glEnd()
    glEndList()
    return gl_pc_list

def draw_sub_verts():
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    size = 1
    sca = (1 / 152.2) * size

    #glBegin(GL_LINES)

    for blob in SUBSECTOR_ARRAY:
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        vg = Vector3(SUBSECTOR_ARRAY[blob]['z_center'])
        glVertex3f(vg[0], vg[1], vg[2])
        vg *= 1.3
        #glVertex3f(0.0,0.0,0.0)
        glVertex3f(vg[0], vg[1], vg[2])
        glEnd()

        for sb in SUBSECTOR_ARRAY[blob]['sectors']:
            glColor3f(1.0, 0.0, 1.0)
            glBegin(GL_LINES)
            sub_blob = SUBSECTOR_ARRAY[blob]['sectors'][sb]
            vg = Vector3(sub_blob['z_center'])
            glVertex3f(vg[0], vg[1], vg[2])
            vg *= 1.2
            #glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(vg[0], vg[1], vg[2])
            glEnd()

            chars = sub_blob['id']
            glPushMatrix()
            glColor3f(1.0, 1.0, 1.0)

            glTranslate(vg[0], vg[1], vg[2])
            glScalef(sca, sca, sca)
            for v in str(chars):
                glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN, ord(v))
            glPopMatrix()

            #
            # a = (sub_blob['v'][0]*1.01).as_tuple()
            # b = (sub_blob['v'][1]*1.01).as_tuple()
            # c = (sub_blob['v'][2]*1.01).as_tuple()
            #
            # glVertex3fv(a)
            # glVertex3fv(b)
            #
            # glVertex3fv(b)
            # glVertex3fv(c)
            #
            # glVertex3fv(c)
            # glVertex3fv(a)




    #glEnd()
    glEndList()
    return gl_sub_list

def draw_poly_hilight_group():
    t = 1.001
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    glColor3f(1.0, 0.3, 1.0)
    glBegin(GL_POINTS)
    for v in POLY_CENTERS_ARRAY:
        v1 = POLY_CENTERS_ARRAY[v]['center']
        if BOX.testBounds(v1):
            glVertex3f(v1.x, v1.y, v1.z)
    glEnd()

    #glFrontFace(GL_CCW)
    #glVertex3f(v1.x, v1.y, v1.z)
    # vertices, normals, texture_coords, material = obj.faces[v]
    # a = (Vector3(obj.vertices[vertices[0]-1])*t).as_tuple()
    # b = (Vector3(obj.vertices[vertices[1]-1])*t).as_tuple()
    # c = (Vector3(obj.vertices[vertices[2]-1])*t).as_tuple()
    # glColor3f(1.0, 0.3, 1.0)
    # glBegin(GL_LINES)
    #
    # glVertex3fv(a)
    # glVertex3fv(b)
    #
    # glVertex3fv(b)
    # glVertex3fv(c)
    #
    # glVertex3fv(c)
    # glVertex3fv(a)
    # glEnd()
    # glBegin(GL_POLYGON)
    # glColor3f(0.3, 0.3, 0.3)
    # glVertex3fv(a.as_tuple())
    # glVertex3fv(b.as_tuple())
    # glVertex3fv(c.as_tuple())
    # glEnd()

    glEndList()
    return gl_sub_list
    pass

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




def pyramid_d():
    pyramid = glGenLists(1)
    glNewList(pyramid, GL_COMPILE)
    glColor4f(0.0, 1.0, 0.1, 1.0)
    h = float(2.28824561127089)
    s = float(1.41421356237309)

    glBegin(GL_POLYGON)
    a = Vector3(0.0, 0.0, s)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(s, 0.0, 0.0)
    N = (a - b).cross(c - b).normalize()
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(-s, 0.0, 0.0)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(0.0, 0.0, s)
    N = (a - b).cross(c - b).normalize()
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(0.0, 0.0, -s)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(-s, 0.0, 0.0)
    N = (a - b).cross(c - b).normalize()
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(s, 0.0, 0.0)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(0.0, 0.0, -s)
    N = (a - b).cross(c - b).normalize()
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    glVertex3f(0.0, 0.0, s)
    glVertex3f(s, 0.0, 0.0)
    glVertex3f(0.0, 0.0, -s)
    glVertex3f(-s, 0.0, 0.0)
    glEnd()

    glEndList()
    return pyramid
pyramid = pyramid_d() #the user icon

def draw_user():
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)
    glColor4f(0.0, 1.0, 0.1, 1.0)
    #glFrontFace(GL_CW)
    glPushMatrix()
    glMultMatrixf(user_matrix.to_opengl())
    # glutWireCube(1.0)
    glScale(0.1, 0.1, 0.1)

    glCallList(pyramid)

    glPopMatrix()
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
    size = 0.15
    sca = (1 / 152.2) * size
    glPushMatrix()
    glColor4f(0.0, 1.0, 0.0, 0.4)
    glLoadIdentity()
    glRotate(180,0.0,1.0,0.0)
    glTranslate(1.0, len(lines)*size, camera_distance*0.75)
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

def draw_surface_user():
    du_gl_list = glGenLists(1)
    glNewList(du_gl_list, GL_COMPILE)
    glPushMatrix()

    #glTranslate(origin.x, origin.y, origin.z)
    glMultMatrixf(user_surface_matrix.to_opengl())

    glColor4f(1.0, 1.0, 0.1, 1.0)
    glutWireCube(2.0)

    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, -4.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(2.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 2.0)
    glEnd()

    glPopMatrix()
    glEndList()
    return du_gl_list

def draw_stars():
    star_gl_list = glGenLists(1)
    glNewList(star_gl_list, GL_COMPILE)
    glColor3f(1.0, 1.0, 1.0)

    for n in range(0, len(STAR_ARRAY)):
        #glTranslate(0.0, 0.0, 0.0)
        glPushMatrix()
        #glMultMatrixf(user_matrix.to_opengl())
        origin = STAR_ARRAY[n]
        glTranslate(origin.x, origin.y, origin.z)
        glutSolidSphere(1.0,8,8)
        glPopMatrix()

    #glPopMatrix()
    glEndList()
    return star_gl_list



star_gl_list = draw_stars()


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################



### M O D E L V I E W ###
glMatrixMode(GL_MODELVIEW)
### M O D E L V I E W ###

user_camera_bind = 0
user_jump = 0

sectors = []
user_message = 'WELCOME'
nctr = 0
P_N = Vector3() #previous surface normal
MP = Vector3() #not midpoint vector
surface_normal_tween = 0
gravity_constant_number = 0.001
world_center = Vector3(0.0,0.0,0.0)

user_pos_vector = user_matrix.get_row_vec3(3)
USER.set(user_pos_vector) #Vector3(0.0,0.0,0.0))

ATT_RAD = 0


while 1:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    time_passed = clock.tick(30)
    time_passed_seconds = float(time_passed) / 1000.0
    #print time_passed_seconds

    pressed = pygame.key.get_pressed()
    rotation_direction.set(0.0, 0.0, 0.0)
    movement_direction.set(0.0, 0.0, 0.0)

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
    


    keydown = 0
    if 1 in pressed:
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
            rotation_direction.z = +rs
        elif pressed[K_e]:
            rotation_direction.z = -rs
        if pressed[K_w]:
            movement_direction.z = -ms
        elif pressed[K_s]:
            movement_direction.z = +ms
        if pressed[K_a]:
            movement_direction.x = -ms
        elif pressed[K_d]:
            movement_direction.x = +ms

        if pressed[K_f]:
            SMOKES.gimme_a_fucking_smoke(USER.Pos.copy(),4)

        if pressed[K_x]:
            USER.set_deccel()
            SMOKES.gimme_a_fucking_smoke(USER.Pos.copy(),1)

        if pressed[K_SPACE]:
            user_jump = 1


        keydown = 1



    glLoadMatrixd(camera_matrix.get_inverse().to_opengl())


    user_pos_vector = user_matrix.get_row_vec3(3)
    camera_pos_vector = camera_matrix.get_row_vec3(3)
    camera_up = camera_matrix.get_row_vec3(1)
    #BOX.setClipBounds(camera_pos_vector, user_pos_vector, camera_up) #time_passed_seconds

    if keydown != 0:
        #if user_surface_bind: rotation_direction.z = rotation_direction.y
        # Calculate rotation matrix and multiply by camera matrix
        rotation = rotation_direction * rotation_speed * time_passed_seconds
        rotation_matrix = Matrix44.xyz_rotation(*rotation)
        camera_matrix *= rotation_matrix
        user_matrix *= rotation_matrix
        heading = Vector3(user_matrix.forward)*1
        movement = heading * movement_direction.z * movement_speed
        ship_direction_vector += movement# * time_passed_seconds
        heading_r = Vector3(user_matrix.right)*-1
        movement = heading_r * movement_direction.x * movement_speed
        ship_direction_vector += movement# * time_passed_seconds
        #else:
        #ship_direction_vector*=movement_speed_decay #.set(0.0,0.0,0.0)

    ship_direction_vector *= movement_speed_decay

    USER.set(ship_direction_vector)
    user_message = time_passed_seconds
    user_pos_vector = user_matrix.get_row_vec3(3)
    NPOS = USER.update_pos(user_pos_vector, time_passed_seconds)

    try:
        height_in_sector, sector_normal, sector_poly, sectors, current_sector, sh_err = SHT.locate(world_center, NPOS)
    except TypeError:
        # this fires when the location has an absolutely zero point
        sh_err = 'NoneType Exception'
        sector_normal = Vector3()
        height_in_sector = 0
        current_sector = 0

    if sh_err: user_message = sh_err

    if user_camera_bind == 0:
        #NEED TO SET ORIGIN OF N(SURFACE NORMAL)
        N = camera_matrix.get_row_vec3(1)*-1
    else:
        N = sector_normal.copy().normalize()
    # what if N were set to camera_up?

    surface_position = height_in_sector * NPOS
    dist_from_c = NPOS.get_length()
    dist_from_surface = dist_from_c-surface_position.get_length()

    # N is a big deal. possible to tween from N to N1? yes.
    #this is some big shit right here. # fucking beautiful.

    if N != P_N:
        nctr = 0
        surface_normal_tween = 1
        MP = P_N.copy()
        P_N = N.copy()

    if surface_normal_tween == 1:
        ost = nctr/10.0 #counter by number of frames to iterate
        N = (N*ost + MP*(1-ost)) #.normalize()
        if ost == 1: surface_normal_tween = 0
        nctr += 1

    UF = (user_matrix.get_row_vec3(3)-camera_matrix.get_row_vec3(3)).normalize()
    PN = N.cross(UF).normalize()
    PW = N.cross(PN).normalize()
    #define surface matrix from surface normal.
    #user_surface_matrix = user_matrix.copy()
    user_surface_matrix.set_row(0, PN.as_tuple() )
    user_surface_matrix.set_row(1, (-N).as_tuple() )
    user_surface_matrix.set_row(2, PW.as_tuple() )
    user_surface_matrix.set_row(3, surface_position.as_tuple() )

    #user_matrix = user_surface_matrix

    vel = USER.Velo.length

    ATT_RAD = degrees(acos(USER.Velo.get_normalised().dot(N)))

    if dist_from_surface < 0.1:

        USER.Pos = (surface_position.length + 0.1) * surface_position.copy().normalize()



        if (ATT_RAD > 15.0) and (vel > 4.0):

            USER.bounce(N)
            SMOKES.gimme_a_fucking_smoke(surface_position,10)

        else:
            USER.creep(N)
            USER.Velo *= 0.96

        if user_jump:
            user_jump = 0
            USER.Velo += -N*5.0

    if dist_from_surface < 2.0:
        user_camera_bind = 1
        #glCallList(draw_surface_user())
    else:
        user_camera_bind = 0
        # user_matrix = camera_matrix.copy()
        # 3copy identity rotation from camera, reset translate after

    user_matrix.set_row(3, (USER.Pos.x, USER.Pos.y, USER.Pos.z))

    #



    #
    #
    #
    # #ANG = acos(UF.dot(N))
    # #user_message = str(degrees(ANG))
    #
    # #uvp = user_pos_vector.copy().normalize()
    # #inertial_direction += uvp * gravity_constant_number
    # #inertial_direction /= 1.01#0.99
    #
    # # uvi = inertial_direction.copy().normalize()

    #continue

    CAM_ATT_RAD = 0

    #this is based on rotation matrix as applied to the camera

    U_PLANE_D = 0

    if user_camera_bind == 0:
        """
        aux = self.fpl['nc'] + Y * self.nc_h
        auxld = (aux - self.p).normalize()
        normal = auxld.cross(X)
        self.fru['bottom_d'] = normal.dot(aux)
        self.fru['bottom'] = normal
        self.fpl['btn'] = aux
        """
        #user_matrix.get_row_vec3(3)
        # uxx = user_matrix.get_row_vec3(3) + (2.0 * Vector3(user_matrix.forward))
        # uyy = user_matrix.get_row_vec3(3) + (2.0 * Vector3(user_matrix.up))
        # ulc = user_matrix.get_row_vec3(3)
        # ulc_n = (uxx-ulc).cross(uyy-ulc)
        # uld = ulc_n.dot(user_matrix.get_row_vec3(3))
        #
        # cam_dz = camera_matrix.get_row_vec3(3) + (2.0 * Vector3(camera_matrix.forward))
        #
        # U_PLANE_D = 1.0 #?cmp(ulc_n.dot(cam_dz) - uld,0)



        direc = user_matrix.get_row_vec3(3).normalize()#Vector3(camera_matrix.forward)
        #camera_matrix = camera_matrix.make_identity()
        arv = (camera_matrix.get_row_vec3(2)).normalize()

        v = arv.dot(direc)  #-(90.0*(pi/180.0))
        CAM_ATT_RAD = degrees(asin(v))
        #da = Vector3(0.0, 1.0, 0.0)
        #ax = arv.cross(direc)

        #camera_matrix.make_rotation_about_axis(-1*Vector3(user_matrix.up), (v))

        u = user_matrix.get_row_vec3(3) + (direc * camera_distance)
        camera_matrix.set_row(3, u.as_tuple())




    else:
        direc = Vector3(camera_matrix.forward)

        u = user_matrix.get_row_vec3(3) + (direc * camera_distance)
        camera_matrix.set_row(3, u.as_tuple())

        pass
        #
        # arv = (user_matrix.get_row_vec3(3)-camera_matrix.get_row_vec3(3)).normalize()
        # direc = Vector3(camera_matrix.forward).normalize()
        # #userd = user_matrix.get_row_vec3(3).normalize()#.get_normalized()
        # #dcent = camera_matrix.get_row_vec3(3).copy().normalize()#Vector3(0.0,0.0,0.0)
        #
        # #camd = direc.copy().normalize()#.get_normalized()
        #
        # v = arv.dot(direc) #-(90.0*(pi/180.0))
        # ax = direc.cross(arv)
        #
        # CAM_ATT_RAD = degrees(acos(v))

        #da = camera_matrix.get_row_vec3(3)#.normalize()
        #
        #da = Vector3(1.0, 1.0, 1.0)

        # camera_matrix.make_rotation_about_axis(ax, v)
        #
        # camera_matrix.set_row(3, u.as_tuple())

    #
    #     #direc = Vector3(camera_matrix.forward)
    #     da = user_matrix.get_row_vec3(1).normalize()
    #     camera_matrix.make_rotation_about_axis(da, v)
    #
    #     u = user_matrix.get_row_vec3(3) + (userd * camera_distance)
    #     camera_matrix.set_row(3, u.as_tuple())
    #
    #
    #
    #     #
    #     # cam = direc.get_normalised()
    #     # prt = cam-user_matrix.get_row_vec3(3).normalize()
    #     # #cam = direc.get_normalised()
    #     #
    #     # v = cam.dot(prt)
    #     # da = prt.cross(cam) * -1
    #     # #direc = Vector3(camera_matrix.forward)
    #     #
    #
    #     #     direc = user_matrix.get_row_vec3(3).normalize()
    #     #
    #     #     u = user_matrix.get_row_vec3(3) + (direc * camera_distance)
    #     #
    #     #     #camera_matrix = user_matrix.copy() #.make_rotation_about_axis(da, v)
    #     #     #camera_matrix = user_matrix
    #     #     camera_matrix.set_row(3, u.as_tuple())
    #     #
    #     #
    #     #
    #
    #
    # else:





    #


    light_d = USER.Pos * 1.1
    glLightfv(GL_LIGHT0, GL_POSITION,  (0, 200, 0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))


    glCallList(draw_sector_poly_hilight(obj, current_sector))
    glCallList(TRACE.draw(user_matrix))
    #glCallList(draw_surface_user())
    #glCallList(draw_poly_hilight_group())


    glCallList(obj.gl_list)
    glCallList(star_gl_list)

    #glCallList(subobj.gl_list)
    #glCallList(draw_sub_verts())
    #glCallList(BOX.showBounds())


    #draw_stars()

    #glCallList(draw_subsector_poly_centers())

    glCallList(draw_line(user_matrix, surface_position))
    glCallList(draw_user())
    glCallList(SMOKES.show())

    #glCallList(draw_ship(inertial_direction))


    debug(['DC:' + str(dist_from_c)[0:4],
           'DS:' + str(dist_from_surface)[0:4],
           'sector:' + str(sectors)+' poly:'+str(current_sector),
           'inert:' + str(USER.Velo.length)[0:6],
           'att:' + str(CAM_ATT_RAD)[0:6],
           'arf:' + str(U_PLANE_D)[0:6],
           'msg:' + str(user_message),
           'box:' + SHT.message])



    pygame.display.flip()
    
# import sys, pygame
# from math import radians
# from math import pi
# from pygame.locals import *
from pygame.constants import *
# from pygame.font import Font
# from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL import GLUT
 
from gameobjects.matrix44 import *
from gameobjects.vector3 import *

import random
import plistlib

#from OpenGL.GLUT import 


# IMPORT OBJECT LOADER
from objloader import *

DOC_ROOT = os.path.dirname(os.path.realpath(__file__))







def init_runtime_vars():
    vars_plist = DOC_ROOT + '/runtime-variables.plist'
    print([__file__,'loaded refreshruntimevars'])
    pl = plistlib.readPlist(vars_plist)
    for lvar in pl:
        globals()[lvar] = pl[lvar]

init_runtime_vars()


pygame.init()
pygame.mixer.init()

snd_ping = pygame.mixer.Sound("audio/flame.wav")
snd_ping.set_volume(0.3)


snd_explode = pygame.mixer.Sound("audio/explode-3.wav")


snd_welcome = pygame.mixer.Sound("audio/02 Nasty Spell_3_bip.wav")#aaah.aiff
snd_thud = pygame.mixer.Sound("audio/tonethud2.wav")
snd_beep = pygame.mixer.Sound("audio/beep_2.wav")

snd_jets2 = pygame.mixer.Sound("audio/jets2.wav")
snd_jets2.set_volume(0.4)
snd_jets = pygame.mixer.Sound("audio/jets_bip.wav")
snd_jets.set_volume(0.6)

#Sound("audio/rediculous.aif")
snd_ambient = pygame.mixer.Sound("audio/ambient.wav")
snd_ambient.set_volume(0.4)
snd_ambient.play(-1)
snd_ambient2 = pygame.mixer.Sound("audio/ambient2.wav")
snd_ambient2.set_volume(0.2)
snd_ambient2.play(-1)
#ping.play()

clock = pygame.time.Clock()
viewport = (1200,800)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glEnable(GL_LIGHTING)
#glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, True)

glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))
# ?glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.48, 0.48, 0.48, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))


glEnable(GL_LIGHT1)
glLightfv(GL_LIGHT1, GL_POSITION,  (0, 0, 0, 1.0))
glLightfv(GL_LIGHT1, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0)) #(0.1, 0.2, 0.1, 1.0))
glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 1.0, 0.5, 1.0)) #(0.1, 0.95, 0.1, 1.0))
glLightfv(GL_LIGHT1, GL_SPECULAR, (0.5, 1.0, 0.5, 1.0)) #(0.5, 1.0, 0.5, 1.0))
glLightfv(GL_LIGHT1, GL_CONSTANT_ATTENUATION, (1.0))
glLightfv(GL_LIGHT1, GL_LINEAR_ATTENUATION, (0.1))
glLightfv(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, (0.0125))

#glLightfv(GL_LIGHT1, GL_SPOT_CUTOFF, (5.0))
#glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, (0, 0, 0))

glEnable(GL_POINT_SMOOTH)
glEnable(GL_PROGRAM_POINT_SIZE)

glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)


glShadeModel(GL_SMOOTH)# most obj files expect to be smooth-shaded
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


### P R O J E C T I O N ###
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glFrustum (-1, 1, -1, 1, 2.0, 8.0)
gluLookAt (0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
width, height = viewport
gluPerspective(55, width/float(height), 1, 50.0)
### P R O J E C T I O N ###


obj = OBJ('untitled.obj', swapyz=True)
subobj = OBJ('untitled-sub.obj', swapyz=True)

print(str(len(obj.faces)) + ' faces')
print(str(len(obj.normals)) + ' normals')
print(str(len(obj.vertices)) + ' vertices')

# model_world_radius = 32.0
# camera_distance = 3.0

# camera_matrix = Matrix44()
# user_surface_matrix = Matrix44()
# user_matrix = Matrix44()
# arse_mat = Matrix44()

# user_matrix.translate = (0.00, 0, model_world_radius)#+camera_distance)
# camera_matrix.translate = (0.00, 0, model_world_radius+camera_distance)
# user_matrix.set_row(1,[0.0,-1.0,0.0])

ship_direction_vector = Vector3(0.0,0.0,0.0)
#inertial_direction = Vector3()

rotation_direction = Vector3()
rotation_speed = radians(90.0)
movement_direction = Vector3()

# movement_speed = 0.5
# movement_speed_decay = 0.95

rs = 0.5 #rotation increment
ms = 0.5 #motion increment

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

    def draw(self, env_vect):
        """draws visible tail/trace of positions"""
        if self.trace_counter == self.trace_frame_interval:
            if len(self.vertices) >= self.trace_length_max: self.vertices = self.vertices[2:]
            self.vertices.append((env_vect.as_tuple()))
            self.trace_counter = 0
        else:
            self.trace_counter += 1

        t_list = glGenLists(2)
        glNewList(t_list, GL_COMPILE)
        glPushMatrix()
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
        glPopMatrix()

        glEndList()

        return t_list
    
    def __init__(self):
        self.vertices = []
        self.vertices.append((0.0, 0.0, 0.0))


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
            BOX.setBoundsScale('default')
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


    def test_bounds(self,origin,look):
        #//SEEK TO add mode to enable other types of targets. this is world-polygons only
        self.get_address_from_dict(SUBSECTOR_CENTERS_ARRAY, 'BY_CLIP_BOUNDS')
        final_set = []
        aps = []
        for v in self.aux_vert_map:
            dve = obj.faces[v][0]
            va = Vector3(obj.vertices[dve[0] - 1])
            vb = Vector3(obj.vertices[dve[1] - 1])
            vc = Vector3(obj.vertices[dve[2] - 1])
            d = intersect_test(origin, look, va, vb, vc)
            if d:
                final_set.append([d[0],d[1],v])
        if len(final_set):
            for dd in final_set:
                v = POLY_CENTERS_ARRAY[dd[2]]['center']
                d = origin.get_distance_to(v)
                aps.append((d, dd))
            d = min(aps)[1]
            return [d[0], d[1], d[2]]

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
        #wc = Vector3(0.0,0.0,0.0)
        horizon_threshold = -20.0
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
        glColor3f(1.0, 0.0, 0.0)
        glPointSize(6.0)
        glBegin(GL_POINTS)

        for pte in self.fpl:
            pt = self.fpl[pte]
            glVertex3f(pt.x, pt.y, pt.z)
        glEnd()
        glEndList()
        return u_gl_list
        pass

    def testBounds(self, point):
        eset = ['world','far','near','right','left','top','bottom'] #
        for plane in eset:
            dist = self.fru[plane].dot(point)-self.fru[plane+'_d']
            if dist > 0: return 0
        return 1


class Corey_Has_Smokes(object):
    def __init__(self):
        """smokes is an array"""
        self.smokes = []
        self.gl_list = []
        self.smoke_max_size = 0.1

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
                randy = ((n[2]*2.0)*self.smoke_max_size)
                glTranslate(n[0].x+randy.x, n[0].y+randy.y, n[0].z+randy.z)
                glColor4f(0.5, 0.5, 0.5, 1.0)# - n[1]/puff)#1/n[1])
                glutSolidSphere((n[3]+ds)*self.smoke_max_size, 8, 8)
                glPopMatrix()
            n[1] += 1.0

        glEndList()
        return gl_sub_list


class animator(object):
    """normal vector tweener"""
    def __init__(self):
        self.v1 = Vector3()
        self.v2 = Vector3()
        self.vD = Vector3()
        self.frames = float(animation_frames)
        self.ctr = 0.0
        self.queue = 0
        self.leng = 0
        self.animating = 0

    def init(self, current_v, new_v):
        self.v1 = current_v
        self.v2 = new_v
        self.ctr = 0.0
        self.leng = current_v.length if current_v.length > 0 else 1.0
        self.animating = 1
        self.queue = 0

    def anim_frame(self):
        if self.ctr == self.frames:
            self.animating = 0
            return self.v2

        dt = self.ctr / self.frames
        self.vD = (self.v2 * dt + self.v1 * (1.0 - dt))
        self.vD *= (self.leng/self.vD.length) if self.vD.length > 0 else 1.0
        self.ctr += 1.0
        return self.vD.normalize()

    def get_number(self,num):
        return num*(self.ctr / self.frames)


class cameraHandler(object):
    """the main camera view instance"""
    #//camera handler
    def __init__(self, view_distance):
        self.transmat = Matrix44()
        self.rotmat = Matrix44()
        self.mat = Matrix44()
        self.modes = ['follow','world','ground']
        self.mode = self.modes[0]
        self.message = self.mode +" +init."
        self.camera_distance = view_distance
        self.rel_pos = Vector3()    #relative position
        self.position = Vector3()   #cached position
        self.transmat.set_row(3, (0.0, 0.0, (model_world_radius * 3)))
        self.tweener = animator()
        self.y_off = Vector3()
        self.x_off = Vector3()
        self.z_off = Vector3()
        self.position_pad = Vector3()

    def get_view(self):
        #self.mat = self.transmat*self.rotmat
        self.rotmat.set_row(3,self.transmat.get_row(3))
        return self.rotmat.get_inverse_rot_trans().to_opengl()

    def next_mode(self):
        cm = self.modes.index(self.mode)
        cm += 1
        if cm == len(self.modes): cm = 0
        self.set_mode(self.modes[cm])

    def set_mode(self, mode):
        if mode != self.mode: self.tweener.queue = 1
        self.mode = self.message = mode

    def lookat(self, root_look, root_right):
        """applies new camera orientation based upon arg.
        USES vec3 root_look to define point to look at"""
        UF = (root_look - self._pos).normalize()
        PR = (UF.cross(root_right)*-1.0).normalize()
        PN = UF.cross(PR).normalize()
        PW = PN.cross(PR).normalize()
        self.rotmat.set_row(0, PN.as_tuple())
        self.rotmat.set_row(1, PR.as_tuple())
        self.rotmat.set_row(2, PW.as_tuple())
        pass


    def update_pos(self, root_position):
        """applies new camera position based upon arg."""
        self.camera_distance = camera_distance
        i = 0.08
        j = 0.2

        self.x_off += ((Vector3(USER.mat.right) - self.x_off) * i)
        self.y_off += ((Vector3(USER.mat.up) - self.y_off) * i)
        self.z_off += ((Vector3(USER.mat.forward) - self.z_off) * i)

        if self.mode == 'follow':
            position = self.z_off.normalize()
        elif self.mode == 'ground':
            USER.define_mat_from_normal()
            position = self.z_off.normalize()
            position += USER.ground_offset_height * self.y_off.normalize() * 0.25
        elif self.mode == 'world':
            USER.at_normal = Vector3(self.rotmat.up).normalize()
            position = root_position.unit().normalize()

        self.rel_pos = position.normalize() * self.camera_distance
        u = root_position + self.rel_pos
        self.position_pad += (u - self.position_pad)*j

        u_ultimate_pos = self.position_pad if self.mode == 'ground' else u
        self.transmat.set_row(3, u_ultimate_pos.as_tuple())
        self.lookat(root_position, self.x_off.normalize())
        pass


    @property
    def _pos(self):
        return Vector3(self.transmat.translate) #get_row_vec3(3)


class worldPhysics(object):
    """Rudimentary Physics Handler.
    aggregate function for any pysical object in world"""
    #//modify for square of distance...less on when far.
    gravity_c = -9.810*gravity_multiplier

    def __init__(self):
        self.Accel = Vector3()
        self.Pos = Vector3()
        self.OldVelo = Vector3()
        self.Velo = Vector3()
        self.GravityVector = Vector3()
        self.Thrusters = Vector3()
        self.Jumpy = 0

    def update(self, delta_time):
        self.Jumpy = 0
        self.GravityVector = self.Pos.get_normalised() * self.gravity_c
        self.Accel = self.GravityVector + self.Thrusters
        self.OldVelo = self.Velo.copy()
        self.Velo += self.Accel * delta_time
        self.Pos += (self.OldVelo + self.Velo) * 0.5 * delta_time
        return self.Pos

    def set(self, direction_vector):
        self.Thrusters = direction_vector

    def set_position(self, position_vector):
        self.Pos = position_vector

    def set_deccel(self):
        self.Velo *= 0.8
        pass

    def bounce(self, N):
        # par = super(self.__class__,self)
        #print(self.__class__.__mro__[1].name)
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


class PhysicalWorldElement(object):
    """handle everything
    parent class of anything that moves and interacts with material world"""
    #//HANDLE ALL THE THINGS!
    name = ''

    def __init__(self, name):
        self.PHYS = worldPhysics()
        self.SCHD = SectorHandle()
        self.TRAC = Trace()
        self.mat = Matrix44()
        self.at_normal = Vector3()
        self.prev_normal = Vector3()
        self.address = 'address not set'
        self.name = name
        self.state_message = "PhysicalWorldElement:"+str(name)
        self.position = Vector3()
        self.surface_position = Vector3()
        self.altitude = 0.0
        self.zero_altitude = 0.0

    def apply_rotation_to_mat(self, rotation):
        rotation_matrix = Matrix44.xyz_rotation(*rotation)
        self.mat *= rotation_matrix

    def define_mat_from_normal(self):
        self.prev_normal += (self.at_normal - self.prev_normal)*0.05
        PPN = self.prev_normal.normalize()
        UF = Vector3(self.mat.forward).normalize()*-1.0
        PN = UF.cross(PPN).normalize()
        PW = PN.cross(PPN).normalize()
        self.mat.set_row(0, PN.as_tuple())
        self.mat.set_row(1, PPN.as_tuple())
        self.mat.set_row(2, PW.as_tuple())

    def update(self, timepassed):
        self.position = self.PHYS.update(timepassed)
        current_position = self.position

        try:
            s_height, normal, poly, s_sectors, sector, s_err = self.SCHD.locate(world_center,current_position)
            self.address = 'sector: ' + str(s_sectors) + ' poly: ' + str(sector) + ' err: ' + str(s_err)

            NN = normal.normalize()
            self.at_normal = -NN.copy()
            vel = self.PHYS.Velo.length
            ATT_RAD = 0

            self.surface_position = s_height * current_position
            self.zero_altitude = current_position.get_length()
            self.altitude = self.zero_altitude - self.surface_position.get_length()

            if (self.altitude < 0.1) and not self.PHYS.Jumpy:
                try:
                    ATT_RAD = degrees(asin(self.PHYS.Velo.get_normalised().dot(NN)))
                except ValueError as e:
                    self.state_message = str(ATT_RAD, e)
                    pass

                self.PHYS.Pos = (self.surface_position.length + 0.1) * self.surface_position.copy().normalize()

                if (ATT_RAD > 45.0) and (vel > 4.0):
                    print(self.name,"bounced")
                    self.PHYS.bounce(NN)
                    snd_thud.play()
                    SMOKES.gimme_a_fucking_smoke(self.surface_position, 10)
                else:
                    self.PHYS.creep(NN)
                    self.PHYS.Velo *= surface_velocity_decay

        except TypeError:
            self.state_message = str("error caught: no address") # this fires when the location has an absolutely zero point
            pass

    @property
    def _pos(self):
        return self.position

    pass


class EntityHandler(PhysicalWorldElement):
    """instance class for entities"""
    def __init__(self, name):
        super(EntityHandler, self).__init__(name)
        e = random.random()
        rotation_direction = Vector3(0.0, e, 0.0)
        self.apply_rotation_to_mat(rotation_direction)
        pass

    def move_entity(self):
        self.define_mat_from_normal()
        rotation_direction = Vector3(0.0,0.001,0.0)
        self.apply_rotation_to_mat(rotation_direction)
        heading_f = Vector3(self.mat.forward) * -1.0
        movement = heading_f * movement_speed
        movement *= movement_speed_decay * 0.5
        self.PHYS.set(movement)
        pass

    def draw_entity(self):
        pos = self._pos
        u_gl_list = glGenLists(1)
        glNewList(u_gl_list, GL_COMPILE)
        glColor4f(0.1, 0.8, 0.1, 1.0)
        glFrontFace(GL_CW)
        glPushMatrix()
        glTranslate(pos.x, pos.y, pos.z)
        glMultMatrixf(self.mat.to_opengl())
        glutSolidIcosahedron()
        glPopMatrix()
        glEndList()
        return u_gl_list




class UserHandler(PhysicalWorldElement):
    """the main user instance"""
    #//user handler
    def __init__(self, name):
        super(UserHandler, self).__init__(name)
        self.message = "hello world"
        self.targeting = {'targetnormal':Vector3(),'targetsector':0}
        self.ground_offset_height = 0.0

    def jump(self):
        self.PHYS.Jumpy = 1
        self.PHYS.Velo += self.at_normal * 2.0

    def apply_rotation(self, rotation):
        if CAM.mode == 'ground':
            self.ground_offset_height += rotation[0]
            if self.ground_offset_height < 0: self.ground_offset_height = 0.0
            rotation[0] *= 0.0
        super(UserHandler, self).apply_rotation_to_mat(rotation)









class Flame(object):
    def __init__(self):
        self.flames_max = 60
        self.flames = []
        self.firetimer = 0
        self.fireinterval = 5
        pass

    def fire(self, origin, direction_v, known_target, target_distance):
        if self.firetimer == 0:
            self.flames.append([origin.copy(), origin.copy(), direction_v, known_target, target_distance])
            self.firetimer = self.fireinterval
            USER.PHYS.Velo += direction_v*0.2
            snd_ping.stop()
            snd_ping.play()

    def show(self):
        if self.firetimer > 0: self.firetimer -= 1

        gl_fire_list = glGenLists(2)
        glNewList(gl_fire_list, GL_COMPILE)
        glFrontFace(GL_CW)

        for flam in self.flames:
            #print flam
            flam[1] -= flam[2]*1.0
            #print((flam[0]-flam[1]).length)

            if ((flam[1]-flam[0]).length) >= floor(abs(flam[4])):
                #end here
                snd_explode.set_volume(1.0/(abs(flam[4])/5.0))
                snd_explode.stop()
                snd_explode.play()
                SMOKES.gimme_a_fucking_smoke(flam[1],40)
                #print(len(self.flames),((flam[1]-flam[0]).length),floor(abs(flam[4])))
                glPushMatrix()
                glColor4f(1.0, 1.0, 0.0, 0.5)
                glTranslate(flam[1].x, flam[1].y, flam[1].z)
                glutSolidSphere(1.2,8,8)
                glPopMatrix()
                self.flames.remove(flam)
            else:


                glPushMatrix()

                glTranslate(flam[1].x, flam[1].y, flam[1].z)
                glColor4f(1.0, 1.0, 0.0, 1.0)
                glScale(0.1,0.1,0.1)

                glutSolidIcosahedron()

                glPopMatrix()

        glEndList()
        return gl_fire_list





        pass



SMOKES = Corey_Has_Smokes()

#if TRACE_ENABLE: TRACE = Trace()

CAM = cameraHandler(camera_distance)

USER = UserHandler('user')
USER.PHYS.set_position(Vector3(0.0, 0.0, float(model_world_radius)*1.2))

# BLOB = EntityHandler('entity-one:castelsnatch')
# BLOB.PHYS.set_position(Vector3(0.0, float(model_world_radius)*2.2, float(model_world_radius)*2.2))

BLOBS = []

for l in range(0,3):
    #//find definitive way to set start position in light of worldPhysics class
    e = random.random()#+1.0
    F = EntityHandler('entity-one:castelsnatch:'+str(l))
    F.PHYS.set_position(Vector3(float(model_world_radius)*e, float(model_world_radius)*e, float(model_world_radius)*e))
    F.PHYS.update(0.1)
    BLOBS.append(F)


#USER.PHYS.update(0.01)

FLAM = Flame()


#OH GOD IT WORKS NOW
N = camera_distance
F = camera_distance*10

BOX = ViewClipBox(N, F, float(width)/(F), float(height)/(F), 0.1)

##################################################################################################
##################################################################################################
############################## W O R L D - M O D E L - D R A W ###################################
##################################################################################################
##################################################################################################

STAR_ARRAY = []
POLY_CENTERS_ARRAY = {}
SUBSECTOR_ARRAY = {}
SUBSECTOR_CENTERS_ARRAY = []
STATIC_BLOBS_ARRAY = []




def build_stars(howmany):
    m = 2.0
    w = howmany*10.0
    for n in range(0,howmany):
        a = random.random()*m - m/2
        b = random.random()*m - m/2
        c = random.random()*m - m/2
        d = random.random()*w
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
        snd_beep.play()
        index += 1


    for ele in SUBSECTOR_CENTERS_ARRAY:
        print(ele)






def make_subdivisions(start_element):
    """unimplemented yet"""
    pass





build_polygon_topography(obj, 'nodal_random', 18)
#build_polygon_topography(obj, 'arbitrary_random', 0.2)
obj.refresh()

build_polygon_centers(obj)

build_sector_addresses(subobj)

build_stars(stars_count)



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
def arrow_d():
    arrow = glGenLists(1)
    glNewList(arrow, GL_COMPILE)
    glColor4f(0.0, 1.0, 0.1, 0.4)
    glFrontFace(GL_CW)

    glBegin(GL_POLYGON)
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, -3.0)
    glVertex3f(-2.5, 0.0, 0.0)
    glVertex3f(-1.0, 0.0, 0.0)
    glVertex3f(-1.0, 0.0, 2.0)
    glVertex3f(1.0, 0.0, 2.0)
    glVertex3f(1.0, 0.0, 0.0)
    glVertex3f(2.5, 0.0, 0.0)
    glVertex3f(0.0, 0.0, -3.0)

    glEnd()
    glEndList()
    return arrow
arrow = arrow_d() #the user arrow-marker icon
def pyramid_d():
    pyramid = glGenLists(1)
    glNewList(pyramid, GL_COMPILE)
    glFrontFace(GL_CCW)

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
    pos = USER._pos
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)
    glColor4f(0.0, 1.0, 0.1, 1.0)

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_arrow_scale, user_arrow_scale, user_arrow_scale)
    glCallList(arrow)
    glPopMatrix()

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_ship_scale, user_ship_scale, user_ship_scale)
    glCallList(pyramid)
    glPopMatrix()






    glEndList()

    return u_gl_list

def draw_surface_user():
    du_gl_list = glGenLists(1)
    glNewList(du_gl_list, GL_COMPILE)
    pos = USER._pos

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())

    glColor4f(1.0, 0.1, 0.1, 1.0)
    glutWireCube(1.0)

    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 1.0)
    glEnd()

    glPopMatrix()
    glEndList()
    return du_gl_list

def draw_subsector_poly_centers():
    #POLY_CENTERS_ARRAY
    gl_pc_list = glGenLists(2)
    glNewList(gl_pc_list, GL_COMPILE)
    glPointSize(4.0)
    #glBegin(GL_POINTS)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    #selected_vert_map
    t = 1.001

    for kk in SHT.aux_vert_map:
        #v = POLY_CENTERS_ARRAY[kk]['center'].copy()*1.001
        #glVertex3f(v.x, v.y, v.z)

        vertices, normals, texture_coords, material = obj.faces[kk]
        a = (Vector3(obj.vertices[vertices[0]-1])*t).as_tuple()
        b = (Vector3(obj.vertices[vertices[1]-1])*t).as_tuple()
        c = (Vector3(obj.vertices[vertices[2]-1])*t).as_tuple()
        glColor3f(1.0, 0.3, 1.0)
        glBegin(GL_LINES)

        glVertex3fv(a)
        glVertex3fv(b)

        glVertex3fv(b)
        glVertex3fv(c)

        glVertex3fv(c)
        glVertex3fv(a)
        glEnd()




    #glEnd()
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
    #debug_rel_size = 0.1
    sca = (1 / 152.2) * debug_rel_size
    glPushMatrix()
    glColor4f(0.0, 1.0, 0.0, 0.4)
    glLoadIdentity()
    glRotate(180,0.0,1.0,0.0)
    glTranslate(1.0, len(lines)*debug_rel_size, camera_distance) #*0.75)
    linect = 0
    for chars in lines:
        glPushMatrix()
        glTranslate(0.0, linect * -debug_rel_size, 0.0)
        glScalef(sca, sca, sca)
        for v in str(chars):
            glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN, ord(v))
        glPopMatrix()
        linect += 1
    glPopMatrix()

def draw_world_user_line(origin, dpos):
    v = origin

    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)
    glPointSize(4.0)
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

def draw_utility_line(origin,a):
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)
    glPushMatrix()
    glTranslate(origin.x, origin.y, origin.z)
    glColor3f(0.5, 1.0, 0.5)

    glBegin(GL_LINES)
    glVertex3f(0.0,0.0,0.0)
    glVertex3f(a.x, a.y, a.z)
    glEnd()

    glPointSize(10.0)
    glBegin(GL_POINTS)

    glVertex3f(a.x, a.y, a.z)
    glEnd()

    glPopMatrix()
    glEndList()

    return a_gl_list

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
#draw stars and save to static glList.

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################



### M O D E L V I E W ###
glMatrixMode(GL_MODELVIEW)
### M O D E L V I E W ###

sectors = []
user_message = 'WELCOME'
world_center = Vector3(0.0,0.0,0.0)

ATT_RAD = 0
cam_counter = 0
key_lock = 0

snd_welcome.play()

while 1:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    time_passed = clock.tick(30) #(30)
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
        elif e.type == KEYUP:
            if e.key == K_r:
                key_lock = 0
            print "key-up"
        elif e.type == 5:
            if e.button == 5:
                camera_distance *= 1.1
            elif e.button == 4:
                camera_distance *= 0.9
            pass
            #button 4 is fwd scroll
            #button 5 is bkw scroll
            # if e.button == 4: zpos = max(0.1, zpos-0.1)
            # elif e.button == 5: zpos += 0.1
            # elif e.button == 1: rotate = True
            # elif e.button == 3: move = True

        elif e.type == 6:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == 4:
            i,j = e.rel
            rotation_direction.x = j*0.2
            rotation_direction.y = i*0.2
            #print(i,j)

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
            movement_direction.z = +ms
        elif pressed[K_s]:
            movement_direction.z = -ms
        if pressed[K_a]:
            movement_direction.x = -ms
        elif pressed[K_d]:
            movement_direction.x = +ms

        if pressed[K_r]:
            #CAMERA MODE SWITCH
            if key_lock == 0:
                snd_beep.play()
                CAM.next_mode()
                key_lock = 1

        if pressed[K_f]:
            SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(),4)

        if pressed[K_x]:
            #snd_jets2.stop()
            snd_jets2.play()
            USER.PHYS.set_deccel()
            SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(),1)

        if pressed[K_c]:
            USER.jump()
            snd_jets.stop()
            snd_jets.play()

        if pressed[K_SPACE]:
            #print USER.targeting
            #know when element hits the target
            #fire_direction = Vector3(USER.mat.forward)
            if USER.targeting:
                #print USER.targeting
                FLAM.fire(USER.targeting['origin'],
                          USER.targeting['direction'],
                          USER.targeting['targetsector'],
                          USER.targeting['disttoimpact'])
            else:
                snd_beep.play()

        keydown = 1


    # user_pos_vector = user_matrix.get_row_vec3(3)
    # camera_pos_vector = camera_matrix.get_row_vec3(3)
    # camera_up = camera_matrix.get_row_vec3(1)




    # Calculate rotation matrix and multiply by camera matrix
    rotation = rotation_direction * rotation_speed * time_passed_seconds
    USER.apply_rotation(rotation)

    if keydown != 0:
        heading = Vector3(USER.mat.forward)*-1.0
        movement = heading * movement_direction.z * movement_speed
        ship_direction_vector += movement # * time_passed_seconds

        heading_r = Vector3(USER.mat.right)*-1.0
        movement = heading_r * movement_direction.x * movement_speed
        ship_direction_vector += movement # * time_passed_second

    ship_direction_vector *= movement_speed_decay
    #user_message = time_passed_seconds





    #NPOS = USER._pos #PHYS.update_pos(time_passed_seconds)
    #USER.mat.translate = NPOS.as_tuple()





    #//THIS ENTIRE set of behaviors needs to be on class instance
    #
    # try:
    #     height_in_sector, sector_normal, sector_poly, sectors, current_sector, sh_err = SHT.locate(world_center, NPOS)
    # except TypeError:
    #     # this fires when the location has an absolutely zero point
    #     print "error caught"
    #     sh_err = 'NoneType Exception'
    #     sector_normal = Vector3()
    #     height_in_sector = 0
    #     current_sector = 0
    #
    #
    # if sh_err: user_message = sh_err
    #
    # # if user_camera_bind == 0:
    # #     #NEED TO SET ORIGIN OF N(SURFACE NORMAL)
    # #     N = camera_matrix.get_row_vec3(1) * -1
    # #     # what if N were set to camera_up?
    # # else:
    # #     N = sector_normal.copy().normalize()
    #
    # #N = sector_normal.copy().normalize()
    #
    # surface_position = height_in_sector * NPOS
    # dist_from_c = NPOS.get_length()
    # dist_from_surface = dist_from_c-surface_position.get_length()
    #
    # # N is a big deal. possible to tween from N to N1? yes.
    # # this is some big shit right here. # fucking beautiful.
    # #
    #
    # #user_surface_matrix.set_row(3, surface_position.as_tuple() )
    # #
    # # #
    #
    # vel = USER.PHYS.Velo.length
    # NN = sector_normal.copy().normalize()
    #
    # # USER.measure()
    # # USER._mpos.unit().normalize(),Vector3(USER.mat.forward),Vector3(USER.mat.up),Vector3(USER.mat.right))
    #
    # if dist_from_surface < 0.1:
    #     if not USER.PHYS.Jumpy:
    #         try:
    #             ATT_RAD = degrees(asin(USER.PHYS.Velo.get_normalised().dot(NN)))
    #         except ValueError as e:
    #             print(ATT_RAD,e)
    #             pass
    #
    #         USER.PHYS.Pos = (surface_position.length + 0.1) * surface_position.copy().normalize()
    #
    #         if (ATT_RAD > 45.0) and (vel > 4.0):
    #             #print("user bounced")
    #             USER.PHYS.bounce(NN)
    #             snd_thud.play()
    #             SMOKES.gimme_a_fucking_smoke(surface_position,10)
    #         else:
    #             USER.PHYS.creep(NN)
    #             USER.PHYS.Velo *= surface_velocity_decay
    #
    #             #USER.mat = user_surface_matrix
    #




    if USER.altitude > 16.0:
        #CAM.set_mode('follow')
        #USER.at_normal = Vector3(USER.mat.up).normalize()

        pass
    else:
        #USER.at_normal = NN * -1
        CAM.set_mode('ground')

    # if CAM.mode == 'world':
    #     WN = Vector3(USER.mat.forward).normalize().cross(USER.PHYS.Velo.unit().normalize()).normalize()
    #     USER.at_normal = WN


    # if dist_from_surface < 3.0:
    #     CAM.set_mode('follow')
    #     # glCallList(draw_surface_user())
    # else:
    #     CAM.set_mode('world')
    #     # USER.mat = camera_matrix.copy()
    #     # 3copy identity rotation from camera, reset translate after
    #
    #     #

    #USER.mat.set_row(3, (USER._pos.x, USER._pos.y, USER._pos.z))


    # CAM_ATT_RAD = 0
    # U_PLANE_D = 0
    #
    # N_look = USER._pos + Vector3(USER.mat.forward) * -1.0 #camera_distance
    # BOX.setClipBounds(USER._pos, N_look, Vector3(USER.mat.up))
    # target = SHT.test_bounds(USER._pos, Vector3(USER.mat.forward))
    #
    # if target:
    #     USER.targeting['origin'] = USER._pos
    #     USER.targeting['direction'] = Vector3(USER.mat.forward)
    #     USER.targeting['disttoimpact'] = target[0]
    #     USER.targeting['targetnormal'] = target[1]
    #     USER.targeting['targetsector'] = target[2]
    # else:
    #     USER.targeting = {}

    ##################################################################################################
    ##################################################################################################

    light_d = USER._pos + Vector3(USER.mat.up)*4.0 #PHYS.Pos * 1.1

    glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))

    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))




    USER.PHYS.set(ship_direction_vector)
    USER.update(time_passed_seconds)
    #print BLOBS

    for BLOB in BLOBS:
        #print BLOB
        BLOB.update(time_passed_seconds)
        BLOB.move_entity()
        glCallList(BLOB.draw_entity())
        glCallList(BLOB.TRAC.draw(BLOB._pos))





    #if target: glCallList(draw_sector_poly_hilight(obj, target[2]))

    #glCallList(draw_sector_poly_hilight(obj, current_sector))
    #glCallList(draw_surface_user())
    #glCallList(draw_utility_line(USER.PHYS.Pos, CAM.offset_marker))
    #glCallList(draw_utility_line(USER.PHYS.Pos, Vector3(CAM.rotmat.right)))
    #glCallList(draw_utility_line(USER.PHYS.Pos, USER.surface_offset*1.5))


    #glCallList(draw_poly_hilight_group())
    #glCallList(subobj.gl_list)
    #glCallList(draw_sub_verts())
    #glCallList(draw_ship(inertial_direction))

    glCallList(draw_world_user_line(USER._pos, USER.surface_position))
    #glCallList(draw_utility_line(USER.PHYS.Pos, Vector3(USER.mat.forward), Vector3(USER.mat.forward)*-100.0))
    #glCallList(draw_utility_line(USER.PHYS.Pos, Vector3(USER.mat.right), Vector3(USER.mat.right)))
    #glCallList(draw_utility_line(USER.PHYS.Pos, USER.X_indicator, -USER.X_indicator))
    #glCallList(draw_utility_line(USER.PHYS.Pos, USER.Y_indicator, -USER.Y_indicator))
    #glCallList(draw_utility_line(USER.PHYS.Pos, Vector3(USER.mat.forward), -300.0*Vector3(USER.mat.forward)))

    glCallList(obj.gl_list)
    glCallList(star_gl_list)
    #glCallList(draw_subsector_poly_centers())
    #glCallList(BOX.showBounds())



    if TRACE_ENABLE:
        glCallList(USER.TRAC.draw(USER._pos))




    glCallList(FLAM.show())
    glCallList(SMOKES.show())
    glCallList(draw_user())


    CAM.update_pos(USER._pos)
    glLoadMatrixf(CAM.get_view())


    # debug(['DC:' + str(dist_from_c)[0:4],
    #        'DS:' + str(dist_from_surface)[0:4],
    #        'sector:' + str(sectors)+' poly:'+str(current_sector),
    #        'inert:' + str(USER.PHYS.Velo.length)[0:6],
    #        'att:' + str(ATT_RAD)[0:6],
    #        'arf:' + str(U_PLANE_D)[0:6],
    #        'msg:' + str(user_message),
    #        'CAM:' + str(CAM.message),
    #        'USR:' + str(USER.message),
    #        'box:' + SHT.message])


    debug(['USR  ' + str(USER.message),
           'ALT  ' + str(USER.altitude)[0:5],
           'STA  ' + str(USER.state_message),
           'ADD  ' + str(USER.address),
           'CAM  ' + str(CAM.message)])






    pygame.display.flip()
    
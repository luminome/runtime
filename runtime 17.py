import random
import plistlib
import os,sys,time
import syslog
import threading
from multiprocessing import Process, Queue
import pygame
from pygame.constants import *
from pygame import font
from pygame import image

from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL import GLUT
 
from gameobjects.matrix44 import *
from gameobjects.vector3 import *

pygame.init()
pygame.mixer.init()
pygame.display.set_caption('Space Program x29')

# IMPORT OBJECT LOADER
from objloader import *

syslog.openlog("Python")

if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    BASEDIR = sys._MEIPASS
else:
    # we are running in a normal Python environment
    BASEDIR = os.path.dirname(__file__)



CAM = None

class MainStackTrace(object):

    def __init__(self):
        self.timer = time.time()
        self.StackTraceBlock = []
        self.blocktime = time.time()
        #self.ABS_POS = Vector3(0.0,0.0,camera_distance)

    def __call__(self,args,*kwargs):
        dt = time.time()
        self.current_message = str('{:.10f}'.format(dt-self.timer))+" "+str(args)
        print(str('{:.10f}'.format(dt-self.timer))+"\t"+str(args))
        if kwargs:
            for earg in kwargs:
                print(str('{:.10f}'.format(dt-self.timer))+"\t"+str(earg))
        print "\n"
        self.StackTraceBlock.append(self.current_message)
        self.timer = dt
        syslog.syslog(syslog.LOG_ALERT, self.current_message)

        if CAM:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_text(1.0, str(self.current_message), USER._pos, 0.0)
            pygame.display.flip()

        pygame.time.wait(100)

    def get_time(self):
        dt = time.time()
        self.current_message = str('{:.10f}'.format(dt - self.blocktime)) + "\tSeconds Elapsed"
        print(self.current_message)
        self.blocktime = dt

StackTrace = MainStackTrace()

def init_runtime_vars(BASEDIR):
    """load runtime variables"""
    vars_plist = os.path.join(BASEDIR, 'runtime-variables.plist')
    StackTrace('loaded refreshruntimevars',vars_plist)
    pl = plistlib.readPlist(vars_plist)
    for lvar in pl:
        globals()[lvar] = pl[lvar]

init_runtime_vars(BASEDIR)

class TypeHandler(object):
    """handle font and display"""
    def __init__(self):
        font.init()
        if not font.get_init():
            print 'Could not render font.'
            sys.exit(0)
        self.font = font.Font(os.path.join(BASEDIR, 'pf_tempesta_seven.ttf'),18)
        self.char = []
        for c in range(256):
            self.char.append(self.create_character(chr(c)))
        self.char = tuple(self.char)
        self.lw = self.char[ord('0')][1]
        self.lh = self.char[ord('0')][2]

    def create_character(self, s):
        try:
            letter_render = self.font.render(s, 1, (50, 255, 50))
            letter = image.tostring(letter_render, 'RGBA', 1)
            letter_w, letter_h = letter_render.get_size()
        except:
            letter = None
            letter_w = 0
            letter_h = 0
        return (letter, letter_w, letter_h)

    def print_string(self, s, x, y, z):
        s = str(s)
        i = 0
        lx = 0
        length = len(s)
        text_list = glGenLists(1)
        glNewList(text_list, GL_COMPILE)
        while i < length:
            glRasterPos2i(x + lx, y)
            glPixelZoom(z,z)
            ch = self.char[ord(s[i])]
            glDrawPixels(ch[1], ch[2], GL_RGBA, GL_UNSIGNED_BYTE, ch[0])
            lx += ch[1]
            i += 1
        glEndList()
        return (text_list,lx)

TEXT = TypeHandler()

class audio(object):
    audio_lex = {}
    def __call__(self, asnd, *args):
        psnd = self.audio_lex[asnd]
        psnd['sound'].set_volume(0)
        psnd['sound'].stop()
        psnd['sound'].set_volume(psnd['basevolume'])
        if len(args) == 1: psnd['sound'].set_volume(args[0]*psnd['basevolume'])
        repeat = args[1] if len(args) == 2 else 0
        psnd['sound'].play(repeat)

    def add(self, resource, snd_name, volume):
        res = {'basevolume':volume,
               'sound':pygame.mixer.Sound(os.path.join(BASEDIR, str(resource)))}
        res['sound'].set_volume(volume)
        self.audio_lex[snd_name] = res

snd = audio()





#BACKGROUND SOUNDS
snd.add("audio/ambient.wav", 'amb', 0.4)
snd.add("audio/ambient2.wav", 'amb2', 0.2)
#snd.add("audio/rediculous.aif", 'redic', 0.4)
snd('amb', 0.4, -1)
snd('amb2', 0.2, -1)
#snd('redic', 0.2, -1)

snd.add('audio/jets_bip.wav', 'jets', 0.5)
snd.add('audio/tonethud2.wav', 'thud', 0.75)
snd.add('audio/beep_2.wav', 'beep', 0.5)
snd.add('audio/flame.wav', 'fire', 0.3)
snd.add('audio/02 Nasty Spell_3_bip.wav', 'welcome', 0.6)
snd.add('audio/explode-3.wav', 'explode', 0.85)
snd.add('audio/aaah_bip.aif', 'ah', 1.0)
snd.add('audio/impactexplosion.wav', 'impact', 0.5)
snd.add('audio/shot2.aiff', 'hit', 0.5)
snd.add('audio/ping.aif', 'tweak', 0.5)
snd.add('audio/negative.wav', 'nega', 0.5)

snd.add('audio/computerbeep_17_bip_1.aif', 'system-beep', 0.15)
snd.add('audio/computerbeep_56.aif', 'affirm-beep', 0.5)



clock = pygame.time.Clock()
viewport = (1200,800)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glEnable(GL_LIGHTING)
#glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, True)

glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))
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
#glDisable(TEXTURE_2D)

### P R O J E C T I O N ###
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glFrustum (-1, 1, -1, 1, 2.0, 12.0)
gluLookAt (0.0, 0.0, -camera_distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
width, height = viewport
gluPerspective(50, width/float(height), 1.0, 31.0)
### P R O J E C T I O N ###\

syslog.syslog(syslog.LOG_ALERT,'GL params passed')



class NPCSet(object):
    NPCs_LIST = []

    def __init__(self):
        StackTrace("initialized NPC set [class NPCSet]")
        pass

    def generate_blob_entities(self, howmany):
        """the class of red blobby npcs"""
        self.NPCs_LIST = []
        m = 2.0
        for l in range(0, howmany):
            a = random.random() * m - m / 2.0
            b = random.random() * m - m / 2.0
            c = random.random() * m - m / 2.0
            ev = Vector3(a, b, c)
            ENTT = EntityHandler('Entity-00' + str(l))
            ENTT.PHYS.set_position(ev * model_world_radius)
            self.NPCs_LIST.append(ENTT)
        StackTrace("initialized %d NPCs" % howmany)

globals()['world_model_ready_state'] = 0

class WorldModelSet(object):

    obj = {}
    subobj = {}
    POLY_CENTERS_ARRAY = {}
    SUBSECTOR_ARRAY = {}
    SUBSECTOR_CENTERS_ARRAY = []
    SELECTED_APEX_VERTS = []
    STAR_ARRAY = []
    ready_state = 0


    def build_stars(self,howmany):
        m = 2.0
        w = howmany * 10.0
        for n in range(0, howmany):
            a = random.random() * m - m / 2
            b = random.random() * m - m / 2
            c = random.random() * m - m / 2
            d = random.random() * w
            starpos = Vector3(a, b, c) * d
            cc = model_world_radius * 2 * starpos.get_normalised()
            self.STAR_ARRAY.append(starpos + cc)


    def build_polygon_topography(self, method, amt, amto):

        if method is 'nodal_random':
            heightmax = 4.0
            radius = (model_world_radius * pi * 2) / (heightmax * heightmax)

            for e in range(amt):
                self.SELECTED_APEX_VERTS.append(random.randrange(0, len(self.obj.vertices)))

            offsetmax = 1
            offsetmin = 2

            for vertexid in self.SELECTED_APEX_VERTS:
                dl = []
                vS = Vector3(self.obj.vertices[vertexid])
                for vertex in self.obj.vertices:
                    v0 = Vector3(vertex)
                    d = vS.get_distance_to(v0)
                    if d < radius and d > 0:
                        dl.append(d)
                        ed = (radius / d)
                        if ed > offsetmax: offsetmax = ed
                        if ed < offsetmin: offsetmin = ed

                min_d = min(dl)
                #print(min_d,offsetmin,offsetmax)

                index = 0
                for vertex in self.obj.vertices:
                    v0 = Vector3(vertex)
                    d = vS.get_distance_to(v0)
                    if d < radius:
                        if d > 0:
                            ed = (radius / d)
                            nv = ((ed - offsetmin) / (offsetmax - offsetmin)) * heightmax
                        elif d == 0:
                            ed = (radius / min_d)
                            nv = ((ed - offsetmin) / (offsetmax - offsetmin)) * heightmax * 0.9

                        pvertex = v0.copy().normalise() * (v0.length + nv)

                        self.obj.vertices[index] = (pvertex.x, pvertex.y, pvertex.z)

                    index += 1

        if method is 'arbitrary_random':
            scale_offset_max = amto
            index = 0
            for vertex in gon_obj.vertices:
                v0 = Vector3(vertex)
                ct = random.random() * scale_offset_max
                vertex = (v0 * (1 - ct))
                gon_obj.vertices[index] = (vertex[0], vertex[1], vertex[2])
                index += 1
            pass

        index = 0
        for face in self.obj.faces:
            vertices, normals, texture_coords, material = face
            #print(face)
            v0 = Vector3((self.obj.vertices[vertices[0] - 1]))
            v1 = Vector3((self.obj.vertices[vertices[1] - 1]))
            v2 = Vector3((self.obj.vertices[vertices[2] - 1]))
            vA = (v1 - v0)
            vB = (v2 - v0)
            vC = vB.cross(vA)
            vC.normalise()

            self.obj.normals[index] = (vC.x, vC.y, vC.z)
            index += 1

        #return gon_obj
        pass


    def build_polygon_centers(self):
        #now we have two specific models.
        """have vertices for siplified pointset...
        must put higher=res points under that."""

        """construct center-point vertices for all faces.
        populate POLY_CENTERS_ARRAY dictionary in form: k:main_name_int, v:vector
        prints some metrics."""

        index = 0

        for face in self.obj.faces:
            vertices, normals, texture_coords, material = face
            #print(face)
            self.POLY_CENTERS_ARRAY[index] = {}
            self.POLY_CENTERS_ARRAY[index]['hverts'] = []
            v1 = Vector3()
            for ii in range(len(vertices)):
                vg = Vector3((self.obj.vertices[vertices[ii] - 1]))
                self.POLY_CENTERS_ARRAY[index]['hverts'].append(vertices[ii] - 1)
                v1 += vg
            v1 /= 3.0
            self.POLY_CENTERS_ARRAY[index]['center'] = v1

            index += 1

        #print vertex_trace
        #print(POLY_REDUX_ONE)
        fl = len(self.POLY_CENTERS_ARRAY)
        StackTrace("POLY_CENTERS_ARRAY", fl, self.POLY_CENTERS_ARRAY[fl - 1])
        pass


    def build_sector_addresses(self):
        """iterate over faces in submodel icosahedron
        split each face into four sub-polys
        run intersection tests on face and sub-polys
        save this information for addressing
        requires that POLY_CENTERS_ARRAY be set"""
        glColor4f(0.8, 0.8, 0.8, 0.1)
        index = 0

        def zip_face(fv):
            return ((fv[0] / 3.0 + fv[1] / 3.0 + fv[2] / 3.0))

        # trace added verts and destroy
        POLY_CENTERS_ARRAY_SHADOW = []
        POLY_CENTERS_ARRAY_SHADOW.extend(range(0, len(self.POLY_CENTERS_ARRAY) - 1))



        for face in self.subobj.faces:
            self.SUBSECTOR_ARRAY[index] = {}
            dve = face[0]
            va = Vector3(self.subobj.vertices[dve[0] - 1])
            vb = Vector3(self.subobj.vertices[dve[1] - 1])
            vc = Vector3(self.subobj.vertices[dve[2] - 1])

            vam = ((va / 2.0 + vb / 2.0))
            vbm = ((vb / 2.0 + vc / 2.0))
            vcm = ((vc / 2.0 + va / 2.0))

            self.SUBSECTOR_ARRAY[index] = {'id': str(index) + 'iso', 'center': zip_face([va, vb, vc]), 'sectors': {},
                                      'v': [va, vb, vc]}
            self.SUBSECTOR_ARRAY[index]['sectors'] = {
                0: {'id': str(index) + 'a', 'center': zip_face([va, vam, vcm]), 'vertmap': [], 'v': [va, vam, vcm], 'hverts':[]},
                1: {'id': str(index) + 'b', 'center': zip_face([vam, vb, vbm]), 'vertmap': [], 'v': [vam, vb, vbm], 'hverts':[]},
                2: {'id': str(index) + 'c', 'center': zip_face([vbm, vc, vcm]), 'vertmap': [], 'v': [vbm, vc, vcm], 'hverts':[]},
                3: {'id': str(index) + 'd', 'center': zip_face([vam, vbm, vcm]), 'vertmap': [], 'v': [vam, vbm, vcm], 'hverts':[]}}


            for k in self.SUBSECTOR_ARRAY[index]['sectors']:
                registered = []
                registered_verts = []
                ksub = self.SUBSECTOR_ARRAY[index]['sectors'][k]
                va = ksub['v'][0]
                vb = ksub['v'][1]
                vc = ksub['v'][2]
                vertset = []

                for kk in POLY_CENTERS_ARRAY_SHADOW:
                    nil = Vector3(0.0, 0.0, 0.0)
                    v = self.POLY_CENTERS_ARRAY[kk]['center']
                    d = intersect_test(nil, v, va, vb, vc)

                    if d and (d[0] > 0.0):
                        ksub['vertmap'].append(kk)
                        dist = v.copy().normalize().get_distance_to(ksub['center'].copy().normalize())
                        vertset.append([dist, kk, v])
                        for vt in self.POLY_CENTERS_ARRAY[kk]['hverts']:
                            if vt not in registered_verts: registered_verts.append(vt)
                        registered.append(kk)

                ksub['z_center'] = min(vertset)[2]
                if k == 3: self.SUBSECTOR_ARRAY[index]['z_center'] = min(vertset)[2]
                ksub['hverts'] = registered_verts

                sub_c = {'id': ksub['id'],
                         'p': ksub['z_center'],
                         'ref': (index, k)}

                self.SUBSECTOR_CENTERS_ARRAY.append(sub_c)
                POLY_CENTERS_ARRAY_SHADOW = [x for x in POLY_CENTERS_ARRAY_SHADOW if x not in registered]

            StackTrace('building zones for sector '+self.SUBSECTOR_ARRAY[index]['id'])

            snd('system-beep', 0.3)
            index += 1


    def init_model(self,model_file):
        obj_path_one = os.path.join(BASEDIR, model_file)
        return OBJ(BASEDIR, obj_path_one, swapyz=True)


    def init_world_model_bases(self):
        StackTrace("init_world_model_bases")
        self.obj = self.init_model(self.mainmodel)
        self.subobj = self.init_model(self.msubmodel)

        StackTrace("models loaded",
                   os.path.join(BASEDIR, self.mainmodel),
                   os.path.join(BASEDIR, self.msubmodel))

        StackTrace(str(len(self.obj.faces)) + ' faces')
        StackTrace(str(len(self.obj.normals)) + ' normals')
        StackTrace(str(len(self.obj.vertices)) + ' vertices')

        self.build_polygon_centers()
        StackTrace("finished building build_polygon_centers")

        self.build_sector_addresses()
        StackTrace("finished building build_sector_addresses")

        self.build_stars(stars_count)
        StackTrace("finished building build_stars")


    def load_world(self, apex_nodes):
        self.SELECTED_APEX_VERTS = []
        StackTrace("loading world")

        self.build_polygon_topography('nodal_random', apex_nodes, 0.02)
        #build_polygon_topography(obj, 'arbitrary_random', 16, 0.02)
        self.obj.refresh()
        StackTrace("finished build_polygon_topography")

        self.build_polygon_centers()
        StackTrace("finished building build_polygon_centers")

        for vert in self.SELECTED_APEX_VERTS:
            FLAG = Flag()
            FLAG.position = Vector3(self.obj.vertices[vert])

        sta = ("finished adding %d flags" % apex_nodes)

        StackTrace(sta)



    def __init__(self):
        self.mainmodel = 'untitled2.obj'
        self.msubmodel = 'untitled-sub.obj'
        StackTrace("initialized world model set [class WorldModelSet]")
        pass







#END DATA FILE SUBMISSIONS





##################################################################################################
##################################################################################################
####################################### U T I L I T Y ############################################
##################################################################################################
##################################################################################################
#//add arrows for enemies (ENTT)
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
def entity_d(radius):
    entity = glGenLists(1)
    glNewList(entity, GL_COMPILE)
    glFrontFace(GL_CW)
    glutSolidSphere(radius, 12, 12)
    glTranslate(radius *0.4, 0.0, radius * -0.9)
    glutSolidSphere(radius * 0.2, 8, 8)
    glTranslate(radius *-0.8, 0.0, 0.0)
    glutSolidSphere(radius * 0.2, 8, 8)
    glEndList()
    return entity
def target_d(radius):
    target_list = glGenLists(1)
    glNewList(target_list, GL_COMPILE)
    #glScalef(radius,radius,radius)
    glutWireCube(radius*1.2)
    # glBegin(GL_LINES)
    # r = radius*1.2
    # glVertex3f(0.0, 0.0, 0.0)
    # glVertex3f(0.0, r, 0.0)
    # glVertex3f(0.0, -r, 0.0)
    # glVertex3f(0.0, 0.0, 0.0)
    # glVertex3f(r, 0.0, 0.0)
    # glVertex3f(-r, 0.0, 0.0)
    # glVertex3f(0.0, 0.0, 0.0)
    # glVertex3f(0.0, 0.0, r)
    # glVertex3f(0.0, 0.0, -r)
    # glEnd()

    glEndList()
    return target_list
def flag_d():
    flag_list = glGenLists(1)
    glNewList(flag_list, GL_COMPILE)
    glFrontFace(GL_CW)
    glColor4f(1.0, 1.0, 0.1, 1.0)
    h = 1.0
    w = 0.01
    glFrontFace(GL_CW)
    glBegin(GL_POLYGON)
    glNormal(0.0, 0.0, 1.0)
    glVertex3f(0.0, h, w)
    glVertex3f(h * 0.5, h, w)
    glVertex3f(h * 0.4, h * 0.8, w)
    glVertex3f(h * 0.5, h * 0.6, w)
    glVertex3f(0.0, h * 0.6, w)
    glVertex3f(0.0, h, w)
    glEnd()

    glFrontFace(GL_CCW)
    glBegin(GL_POLYGON)
    glNormal(0.0, 0.0, -1.0)
    glVertex3f(0.0, h, -w)
    glVertex3f(h * 0.5, h, -w)
    glVertex3f(h * 0.4, h * 0.8, -w)
    glVertex3f(h * 0.5, h * 0.6, -w)
    glVertex3f(0.0, h * 0.6, -w)
    glVertex3f(0.0, h, -w)
    glEnd()

    glFrontFace(GL_CW)
    glBegin(GL_POLYGON)
    glVertex3f(0.0, 0.0, w)
    glVertex3f(0.0, h, w)
    glVertex3f(h * 0.05, h, w)
    glVertex3f(h * 0.05, 0.0, w)
    glVertex3f(0.0, 0.0, w)
    glEnd()

    glFrontFace(GL_CCW)
    glBegin(GL_POLYGON)
    glVertex3f(0.0, 0.0, -w)
    glVertex3f(0.0, h, -w)
    glVertex3f(h * 0.05, h, -w)
    glVertex3f(h * 0.05, 0.0, -w)
    glVertex3f(0.0, 0.0, -w)
    glEnd()


    # glFrontFace(GL_CW)
    # glBegin(GL_POLYGON)
    # glVertex3f(0.0, 0.0, -w)
    # glVertex3f(0.0, 0.0, w)
    # glVertex3f(0.0, h, w)
    # glVertex3f(0.0, h, -w)
    # glVertex3f(0.0, 0.0, -w)
    # glEnd()
    #
    # glFrontFace(GL_CCW)
    # glBegin(GL_POLYGON)
    # glVertex3f(h * 0.05, 0.0, -w)
    # glVertex3f(h * 0.05, 0.0, w)
    # glVertex3f(h * 0.05, h, w)
    # glVertex3f(h * 0.05, h, -w)
    # glVertex3f(h * 0.05, 0.0, -w)
    # glEnd()
    #
    #
    #

    glEndList()
    return flag_list


def target_acquired(entity):
    pos = entity._pos
    et = target_d(entity.radius*2.0)

    target_acquired_list = glGenLists(1)
    glNewList(target_acquired_list, GL_COMPILE)
    glColor4f(0.1, 1.0, 0.1, 1.0)

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(entity.mat.to_opengl())

    glCallList(et)
    glPopMatrix()

    glEndList()
    return target_acquired_list


def get_prox(mat,a,b):
    pdx = (b - a)
    urw = Vector3(mat.right).normalize()
    ups = Vector3(mat.up).normalize()
    hps = pdx.cross(ups).normalize()
    angle = acos(urw.dot(hps))
    scross = urw.cross(hps)
    if ups.dot(scross) < 0: angle = -angle
    return angle

    # pdx = (b - a)
    # urw = Vector3(mat.right).normalize()
    # ups = Vector3(mat.up).normalize()
    # hps = pdx.cross(ups).normalize()
    # angle = acos(dot(urw, hps))
    # scross = cross(urw, hps)
    # if dot(ups, scross) < 0: angle = -angle
    # return angle


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
            v = WORLD.POLY_CENTERS_ARRAY[kk]['center'].copy().normalize()
            d = self.seek_vector_normal.get_distance_to(v)
            if d < scope_normalized:
                vert_set.append(kk)
        if len(vert_set) == 0: return 0

        for v in vert_set:
            dve = WORLD.obj.faces[v][0]
            va = Vector3(WORLD.obj.vertices[dve[0] - 1])
            vb = Vector3(WORLD.obj.vertices[dve[1] - 1])
            vc = Vector3(WORLD.obj.vertices[dve[2] - 1])
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
                    verts += WORLD.SUBSECTOR_ARRAY[kk['ref'][0]]['sectors'][kk['ref'][1]]['vertmap']
            self.message = str((len(verts),'selected'))
            BOX.setBoundsScale('default')
            self.aux_vert_map = [x for x in verts if BOX.testBounds(WORLD.POLY_CENTERS_ARRAY[x]['center'])]
            pass
        else:
            return 0, 'no address'

    def get_address(self):
        """lookup address (recurs)"""
        sel, sector_id = self.get_address_from_dict(WORLD.SUBSECTOR_ARRAY,'BY_DISTANCE')
        sel_sub, sub_sector_id = self.get_address_from_dict(WORLD.SUBSECTOR_ARRAY[sel]['sectors'],'BY_INTERSECT')
        self.selected_vert_map = WORLD.SUBSECTOR_ARRAY[sel]['sectors'][sel_sub]['vertmap']
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

        #//whoa mod clip box
        """better to use secondary test for this imo."""
        aux = self.fpl['nc'] + Y * self.nc_h * 0.1
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
        self.smoke_max_size = 0.2
        print("Corey_Has_Smokes.")

    def gimme_a_fucking_smoke(self, position, howmany, *args, **kwargs):

        if args: print(args)

        for l in range(0,howmany):
            a = random.random() - 0.5
            b = random.random() - 0.5
            c = random.random() - 0.5
            s = random.random()
            randy = Vector3(a,b,c)
            mod = 1.0

            if args: mod = args[0]

            if args and len(args) > 1:
                self.smokes.append([position + randy,
                                    50.0,
                                    randy,
                                    0.2,
                                    ceil(s*100.0),
                                    position,
                                    6.0,
                                    [1.0, 1.0, 0.15, 1.0]])
            else:
                self.smokes.append([position + randy * 0.2,
                                    1.0,
                                    randy,
                                    s,
                                    ceil(s * 60.0) * 1.0,
                                    position,
                                    mod,
                                    [0.25, 0.15, 0.15, 1.0]])




    def show(self):
        gl_sub_list = glGenLists(2)
        glNewList(gl_sub_list, GL_COMPILE)
        glFrontFace(GL_CW)

        for n in self.smokes:
            ds = sin(pi * (n[1]/n[4]))
            n[0] += (n[2] * ds) * (0.03*n[6])
            if n[0].length < n[5].length: n[0] = n[0].get_normalized()*n[5].length
            glPushMatrix()
            glTranslate(n[0].x, n[0].y, n[0].z)
            glColor4f(n[7][0],n[7][1],n[7][2],n[7][3]) #, 0.15, 0.15, 1.0)
            glutSolidSphere((n[3]*ds)*self.smoke_max_size, 8, 8)
            glPopMatrix()
            n[1] += 1.0

        for n in self.smokes:
            if n[1] > n[4]-1.0: self.smokes.remove(n)


        glEndList()
        return gl_sub_list


class Flame(object):
    def __init__(self):
        self.flames_max = 60
        self.flames = []
        self.firetimer = 0
        self.fireinterval = 5
        pass

    def fire(self, origin, direction_v, target_distance, ptarget):
        if self.firetimer == 0:
            self.flames.append([origin.copy(), origin.copy(), direction_v, target_distance, ptarget])
            self.firetimer = self.fireinterval
            USER.PHYS.Velo += direction_v*0.4



    def show(self):
        if self.firetimer > 0: self.firetimer -= 1
        gl_fire_list = glGenLists(2)
        glNewList(gl_fire_list, GL_COMPILE)

        for flam in self.flames:
            d = (flam[1]-flam[4]._pos)
            flam[1] -= d.get_normalized()*0.2

            if (d.length < flam[4].radius):
                self.flames.remove(flam)
                flam[4].flamed([flam[2],flam[3]])
                flam[4].hitting = 1

            else:
                glPushMatrix()
                glTranslate(flam[1].x, flam[1].y, flam[1].z)
                glColor4f(1.0, 1.0, 0.0, 1.0)
                glutSolidSphere(0.025,10,10)
                glPopMatrix()

            if flam[1].length < flam[4]._pos.length: flam[1] = flam[0].length * flam[1].get_normalized()

        glEndList()
        return gl_fire_list





        pass


class CameraHandler(object):
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
        self.transmat.set_row(3, (0.0, 0.0, (model_world_radius * 2.0)))
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
        #if mode != self.mode: self.tweener.queue = 1
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
        i = 0.1
        j = 0.1

        self.x_off += ((Vector3(USER.mat.right) - self.x_off) * i)
        self.y_off += ((Vector3(USER.mat.up) - self.y_off) * i)
        self.z_off += ((Vector3(USER.mat.forward) - self.z_off) * i)

        if self.mode == 'follow':
            position = self.z_off.normalize()
        elif self.mode == 'ground':
            USER.define_mat_from_normal()
            position = self.z_off.normalize()
            position += USER.ground_offset_height * self.y_off.normalize() * 0.1
        elif self.mode == 'world':
            #//STILL PROBLEMATIC
            USER.at_normal = Vector3(self.rotmat.up).normalize()
            USER.define_mat_from_normal()
            position = root_position.unit().normalize()

        self.rel_pos = position.normalize() * self.camera_distance
        u = root_position + self.rel_pos
        self.position_pad += (u - self.position_pad)*j

        u_ultimate_pos = u #self.position_pad if self.mode == 'ground' else u
        self.transmat.set_row(3, u_ultimate_pos.as_tuple())
        self.lookat(root_position, self.x_off.normalize())
        pass


    @property
    def _pos(self):
        return Vector3(self.transmat.translate) #get_row_vec3(3)


class WorldPhysics(object):
    """Rudimentary Physics Handler.
    aggregate function for any pysical object in world"""
    #//modify for square of distance...less on when far.
    gravity_c = -9.810*gravity_multiplier

    def __init__(self):
        self.Accel = Vector3()
        self.Pos = Vector3()
        self.NextPos = Vector3()
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
        self.NextPos = self.Pos + (self.OldVelo + self.Velo) * 0.5 * delta_time
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
        self.Velo = ((-I.dot(N)) * N + I)
        pass


class PhysicalWorldElement(object):
    """handle everything
    parent class of anything that moves and interacts with material world"""
    #//HANDLE ALL THE THINGS!
    name = ''
    PWElements = []

    def __init__(self, name, radius):
        self.PHYS = WorldPhysics()
        self.SCHD = SectorHandle()
        self.TRAC = Trace()
        self.mat = Matrix44()
        self.at_normal = Vector3()
        self.prev_normal = Vector3()
        self.address = 'address not set'
        self.name = name
        self.radius = radius
        self.init_radius = radius
        self.state_message = "PhysicalWorldElement:"+str(name)
        self.position = Vector3()
        self.surface_position = Vector3()
        self.altitude = 0.0
        self.zero_altitude = 0.0
        self.collided = 0
        self.hitting = 0
        self.flaming = 0
        self.killed = 0
        self.PWElements.append(self)
        print("Created PhysicalWorldElement: ", self.name)

    #//FLAMED AN HIT ARE HERE BECAUSE OF EVENTUAL MULTIPLE ENTITY-TYPES
    def flamed(self, by):
        """when the target has been hit by a flame"""
        amt = 0.5
        self.radius *= amt
        self.mass *= amt
        self.PHYS.Velo += by[0]*-1.0
        self.flaming = 1
        snd('impact',(0.75 / by[1]))
        SMOKES.gimme_a_fucking_smoke(self._pos, 10, 1.0, 'fire')


        if self.radius/self.init_radius < 0.1:
            delegated_flames = [f for f in FLAM.flames if f[4].name == self.name]
            for f in delegated_flames:
                #SMOKES.gimme_a_fucking_smoke(f[1], 5, 4.0)
                FLAM.flames.remove(f)

            SMOKES.gimme_a_fucking_smoke(self._pos, 20, 4.0)

            snd('explode', (0.75 / by[1]))
            self.killed = 1
            score(10000, self._pos)
            self.respawn()
        else:
            score(1000, self._pos)
        pass

    def hit(self, sta):
        """when a hit is triggered"""
        if not sta: return

        USER.hitting = 1

        score(100,self._pos)
        snd('tweak',0.15)
        pass

    def visible_output_message(self, string_or_tuple, timedelta):
        self.state_message = (self.name, timedelta, string_or_tuple)
        print(self.state_message)

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

            nn = normal.normalize()
            self.at_normal = -nn.copy()
            vel = self.PHYS.Velo.length


            self.surface_position = s_height * current_position
            self.zero_altitude = current_position.get_length()
            self.altitude = self.zero_altitude - self.surface_position.get_length()

            if (self.altitude < 0.1) and not self.PHYS.Jumpy:
                try:
                    attitude_deg = degrees(asin(self.PHYS.Velo.get_normalised().dot(nn)))
                except ValueError as degree_error:
                    self.visible_output_message((attitude_deg, degree_error),timepassed)
                    #pass

                self.PHYS.Pos = (self.surface_position.length + 0.1) * self.surface_position.copy().normalize()

                if (attitude_deg > 45.0) and (vel > 3.0):
                    #print(self.name,"bounced")
                    self.PHYS.bounce(nn)
                    snd('thud')
                    SMOKES.gimme_a_fucking_smoke(self.surface_position, 10, 2.0)
                else:
                    self.PHYS.Velo *= surface_velocity_decay
                    self.PHYS.creep(nn)


        except TypeError:
            self.visible_output_message("error caught: no address", timepassed)
            #self.state_message = str("error caught: no address") # this fires when the location has an absolutely zero point
            #pass

    @property
    def _pos(self):
        return self.position.copy()

    @property
    def _nextpos(self):
        return self.PHYS.NextPos

    pass


class EntityHandler(PhysicalWorldElement):
    """instance class for entities"""
    #//Entity Handler
    def __init__(self, name):
        er = random.random()*0.5 + 0.5
        super(EntityHandler, self).__init__(name, er)
        aa = int(random.random() * len(names_one))-1
        bb = int(random.random() * len(names_one))-1
        self.real_name = str(names_one[aa] + '-' + names_one[bb])
        names_one.remove(names_one[aa])
        names_one.remove(names_one[bb])
        self.mass = er*10.0
        # self.apply_rotation_to_mat(rotation_direction)
        self.behavior_idle = random.random()*180.0
        self.rotation_amt = 0
        self.spawn_count = 0
        self.PHYS.Velo = Vector3(0.001,0.001,0.001)

    def respawn(self):
        m = 2.0
        er = random.random() * 0.5 + 0.5
        a = random.random() * m - m / 2.0
        b = random.random() * m - m / 2.0
        c = random.random() * m - m / 2.0
        ev = Vector3(a, b, c)
        self.radius = er
        self.init_radius = er
        self.PHYS.set_position(ev * model_world_radius)
        self.killed = 0
        self.spawn_count += 1

    def move_entity(self):
        self.define_mat_from_normal()
        ex = (USER._pos - self._pos)
        salt = (1.0 + (1.0*sin(self.behavior_idle)))
        e_heading = Vector3(self.mat.forward) * -salt
        e_right = Vector3(self.mat.right) * -1.0
        dir_rotate = ex.normalize().dot(e_right)
        rd = Vector3(0.0, dir_rotate + (salt*0.001), 0.0)
        self.apply_rotation_to_mat(rd)

        mov = e_heading * movement_speed
        mov *= movement_speed_decay

        self.PHYS.set(mov)
        self.behavior_idle += 0.5
        #self.rotation_amt += dir_rotate

        pass

    def draw_entity(self):
        pos = self._pos
        ent = entity_d(self.radius)

        u_gl_list = glGenLists(1)
        glNewList(u_gl_list, GL_COMPILE)
        glPushMatrix()
        glTranslate(pos.x, pos.y, pos.z)
        glMultMatrixf(self.mat.to_opengl())

        if self.collided:
            glColor4f(0.7, 0.1, 0.7, 1.0)
            self.collided = 0
        elif self.hitting:
            glColor4f(0.8, 0.8, 0.1, 1.0)
            self.hitting = 0
        else:
            glColor4f(0.5, 0.01, 0.01, 1.0)

        glCallList(ent)
        glPopMatrix()
        glEndList()
        return u_gl_list


class UserHandler(PhysicalWorldElement):
    """the main user instance"""
    #//user handler
    def __init__(self, name):
        super(UserHandler, self).__init__(name, user_ship_scale*1.4)
        self.mass = 150.0
        self.message = "hello world"
        self.targeting = {'targetnormal':Vector3(),'targetsector':0}
        self.ground_offset_height = 0.0
        self.entity_marker_arrow_angles = {}
        self.flag_marker_arrow_angles = {}
        self.score = 0
        self.radius = 1

    def jump(self):
        self.PHYS.Jumpy = 1
        self.PHYS.Velo += self.at_normal * 2.0

    def apply_rotation(self, rotation):
        if CAM.mode == 'ground':
            self.ground_offset_height += rotation[0]
            if self.ground_offset_height < 0: self.ground_offset_height = 0.0
            rotation[0] *= 0.0
        super(UserHandler, self).apply_rotation_to_mat(rotation)


class Flag(object):
    Flags = []
    def __init__(self):
        self.id = len(self.Flags)+1
        self.position = Vector3()
        self.flag_graphic = flag_d()
        self.Flags.append(self)
        self.state = "open"

    def draw_flag(self):
        flag_gl_list = glGenLists(1)
        glNewList(flag_gl_list, GL_COMPILE)

        glPushMatrix()
        glTranslate(self.position.x, self.position.y, self.position.z)
        glColor4f(0.8, 0.8, 0.1, 1.0)

        glPointSize(6.0)
        ep = (self.position * 1.05) - self.position
        glTranslate(0.0,0.0,0.0)
        glBegin(GL_POINTS)
        glVertex3f(ep.x,ep.y,ep.z)
        glEnd()
        glPopMatrix()

        glPushMatrix()
        v1 = Vector3(0.0, 1.0, 0.0)
        v2 = self.position.copy().normalize()
        a_dot = v1.dot(v2)
        angle = acos(a_dot)
        axis = v1.cross(v2).normalize()
        glTranslate(self.position.x, self.position.y, self.position.z)
        glRotate(degrees(angle), axis.x, axis.y, axis.z)
        glCallList(self.flag_graphic)
        glPopMatrix()



        glEndList()
        return flag_gl_list


class ScoreMarker(object):
    scores = []
    class score_e(dict):
        pass

    def __init__(self):
        pass

    def __call__(self, points, pos):
        s = self.score_e()
        s.pos = pos
        s.points = points
        s.frames = 16
        self.scores.append(s)
        pass

    def show(self):
        for s in self.scores:
            s.frames -= 1
            s.pos *= 1.002
            draw_text(1.0, str(s.points), s.pos, 0.0)
        r = [self.scores.remove(s) for s in self.scores if s.frames == 0]

        scpos = CAM._pos + Vector3(CAM.rotmat.forward)*-2.0   #*USER._pos-((USER._pos-CAM._pos)/2.0)
        draw_text(0.5, str(USER.score), scpos, 0.4)

    pass


Score = ScoreMarker()


class Button(object):


    def __init__(self):


        pass



    pass

#//separated functions
def physical_world_element_collisions():
    """handle general collisions on frame
    to avoid redundancies"""
    def collide(clsA, clsB):
        x = (clsA._nextpos - clsB._nextpos)
        if not x.length: return
        mtd = x * (((clsA.radius + clsB.radius) - x.length) / x.length)
        im1 = 1.0 / clsA.mass
        im2 = 1.0 / clsB.mass
        clsA.PHYS.Pos += mtd * (im1 / (im1 + im2))
        clsB.PHYS.Pos -= mtd * (im2 / (im1 + im2))
        vn = (clsA.PHYS.Velo - clsB.PHYS.Velo).dot(mtd.normalize().copy())
        i = ((1.0 + COR)*-vn) / (im1 + im2)
        if vn > 0.0: return
        impulse = mtd * i
        clsA.PHYS.Velo += impulse*im1
        clsB.PHYS.Velo -= impulse*im2


    res = []
    for i in range(0, len(PhysicalWorldElement.PWElements)):
        #trace of collisions with this specific entry
        sel = PhysicalWorldElement.PWElements[i]
        element_collision = [el for el in PhysicalWorldElement.PWElements if (el.name != sel.name) and (el._nextpos.get_distance_to(sel._nextpos) < (sel.radius + el.radius))]

        for el in element_collision:
            if (el,sel) not in res:
                el.collided = 1 if sel.name == 'user' else 0
                sel.collided = 1 if el.name == 'user' else 0

                el.hit(el.collided)
                sel.hit(sel.collided)

                collide(sel,el)
                res.append((sel,el))

def target_entity(class_handle):
    """targeting system for ViewClipBox against entities
    class_handle is the targetor (for now only user)"""
    compare = [ent for ent in PhysicalWorldElement.PWElements if ent.name != class_handle.name]
    ps = []

    for entity in compare:
        has = BOX.testBounds(entity._pos)
        if has: ps.append((entity._pos.get_distance_to(class_handle._pos),entity))

    return min(ps)[1] if len(ps) else 0

def draw_user():
    pos = USER._pos
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_ship_scale, user_ship_scale, user_ship_scale)
    glColor4f(0.2, 0.7, 0.2, 1.0)
    glCallList(pyramid)
    glPopMatrix()

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_arrow_scale, user_arrow_scale, user_arrow_scale)
    glColor4f(0.2, 0.7, 0.2, 1.0)
    glCallList(arrow)
    glPopMatrix()



    for aname, angle in USER.entity_marker_arrow_angles.items():
        glColor4f(0.2, 0.7, 0.2, 1.0)
        siz = 0.025 + 0.05 * (1.0/angle[1])
        glPushMatrix()
        glTranslate(pos.x, pos.y, pos.z)
        glMultMatrixf(USER.mat.to_opengl())
        glRotate(degrees(angle[0]), 0.0, 1.0, 0.0)
        glTranslate(0.0, 0.0, -1.0 + (0.5/angle[1]))
        glScale(siz, siz, siz)
        glCallList(arrow)
        glPopMatrix()

    for aname, angle in USER.flag_marker_arrow_angles.items():
        siz = 10.0*(10.0/angle[1])
        pointsize = siz if siz < 10.0 else 10.0
        glPushMatrix()
        glTranslate(pos.x, pos.y, pos.z)
        glMultMatrixf(USER.mat.to_opengl())
        glPointSize(pointsize)
        glColor4f(1.0, 1.0, 0.0, 1.0)
        glRotate(degrees(angle[0]), 0.0, 1.0, 0.0)
        glTranslate(0.0, 0.0, -2.0 + (2.0/angle[1]))
        glBegin(GL_POINTS)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glPopMatrix()

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glColor4f(0.2, 0.7, 0.2, 0.2)

    if USER.hitting:
        glColor4f(0.2, 0.7, 0.2, 0.6)
        USER.hitting = 0

    glutSolidSphere(USER.radius,16,16)
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

    for kk in USER.SCHD.aux_vert_map:
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

def draw_sub_sector_verts():
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    vert_map = WORLD.SUBSECTOR_ARRAY[USER.SCHD.sector_number]['sectors'][USER.SCHD.sub_sector_number]['hverts']
    glPointSize(2.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.0, 1.0)
    for vert in vert_map:
        vg = WORLD.obj.vertices[vert]
        glVertex3f(vg[0], vg[1], vg[2])
    glEnd()

    glEndList()
    return gl_sub_list

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

    glPushMatrix()
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
    glPopMatrix()

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
    #a -= origin
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
    glColor4f(1.0, 1.0, 1.0, 1.0)

    for n in range(0, len(WORLD.STAR_ARRAY)):
        #glTranslate(0.0, 0.0, 0.0)
        glPushMatrix()
        #glMultMatrixf(user_matrix.to_opengl())
        origin = WORLD.STAR_ARRAY[n]
        glTranslate(origin.x, origin.y, origin.z)
        glutSolidSphere(1.0,8,8)
        glPopMatrix()

    #glPopMatrix()
    glEndList()
    return star_gl_list
star_gl_list = []

def draw_text(scale, string, pos, radius):
    glColor4f(1.0, 1.0, 1.0, 1.0)
    zoom = (float(camera_distance)/CAM._pos.get_distance_to(pos))*scale
    text_list = TEXT.print_string(string, 0, 0, zoom)
    sca2 = (1.0 / 90.0)*scale
    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glScalef(sca2, sca2, sca2)
    glPushMatrix()
    glMultMatrixf(CAM.rotmat.to_opengl())
    glTranslate(text_list[1]/2, 2.0*radius*(152.2/scale), 0.0)
    glRotate(180.0, 0.0, 0.0, 1.0)
    glCallList(text_list[0])
    glPopMatrix()
    glPopMatrix()



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


time_passed = clock.tick()

WORLD = WorldModelSet()
NPCs = NPCSet()

SMOKES = Corey_Has_Smokes()
FLAM = Flame()

CAM = CameraHandler(camera_distance)
USER = UserHandler('user')
uv = Vector3(1.0,1.0,1.0)
USER.PHYS.set_position(uv*model_world_radius)


ship_direction_vector = Vector3()
rotation_direction = Vector3()
#rotation_speed = radians(90.0)
movement_direction = Vector3()
#rs = 1.0 #rotation increment
#ms = 0.5 #motion increment
#rotate = move = False
COR = 1.0 #0.8
#wiki/Coefficient_of_restitution
attitude_deg = 0

N = camera_distance*0.25
F = camera_distance*8
BOX = ViewClipBox(N, F, float(width)/(F), float(height)/(F), 0.1)

ship_direction_vector.set(0.0, 0.0, 0.0)

key_rep_frame = 0
control_vectors = {}
stat_trigger = 0

init_world_size = 2
current_level = 1





def init_blank():
    glMatrixMode(GL_MODELVIEW)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ABS_POS = Vector3(0.0, 0.0, 0.0)
    CAM_POS = Vector3(0.0, 0.0, 0.0)
    CAM.update_pos(CAM_POS)
    glLoadMatrixf(CAM.get_view())
    draw_text(1.0, "loading resources", ABS_POS, 0.0)
    pygame.display.flip()

def x29_first_run():
    """Setup the game"""
    global star_gl_list
    init_blank()
    WORLD.init_world_model_bases()
    star_gl_list = draw_stars()
    snd('welcome')

def x29_world_init(world_size):
    """Setup the game level"""
    USER.entity_marker_arrow_angles = {}
    USER.flag_marker_arrow_angles = {}

    WORLD.load_world(world_size)
    StackTrace('WORLD.load_world(%s) COMPLETE' % world_size)

    e = min([int(world_size/1.25),5])
    NPCs.generate_blob_entities(e)

    StackTrace('NPCs.generate_blob_entities COMPLETE')
    StackTrace(len(WORLD.POLY_CENTERS_ARRAY))
    StackTrace.get_time()


    return 'ok'

def quit_x29():
    StackTrace("quitting")
    pygame.quit()
    exit()


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#//OLD RUNTIME LOOP:
def x29_runtime_set():
    ship_direction_vector = Vector3(0.0, 0.0, 0.0)

    while 1:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        time_passed = clock.tick() #(30)
        time_passed_seconds = float(time_passed) / 1000.0
        pressed = pygame.key.get_pressed()
        rotation_direction.set(0.0, 0.0, 0.0)
        movement_direction.set(0.0, 0.0, 0.0)
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
                rotation_direction.x = j*0.5
                rotation_direction.y = i*0.5
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
                    snd('beep',0.75)
                    CAM.next_mode()
                    key_lock = 1

            if pressed[K_f]:
                SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(),4)

            if pressed[K_v]:
                USER.radius += 0.1
                #SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(),4)

            if pressed[K_x]:
                snd('jets')
                USER.PHYS.set_deccel()
                SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(),1)

            if pressed[K_c]:
                USER.jump()
                snd('jets')
                SMOKES.gimme_a_fucking_smoke(USER.PHYS.Pos.copy(), 5)



            if pressed[K_SPACE]:
                #print USER.targeting
                #know when element hits the target
                #fire_direction = Vector3(USER.mat.forward)
                if USER.targeting:
                    #print USER.targeting
                    FLAM.fire(USER.targeting['origin'],
                              USER.targeting['direction'],
                              USER.targeting['disttoimpact'],
                              USER.targeting['target'])

                    snd('fire',0.6)
                else:
                    snd('beep',0.1)

            keydown = 1


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


        if USER.altitude < 1.0: CAM.set_mode('ground')

        # if CAM.mode == 'world':
        #     WN = Vector3(USER.mat.forward).normalize().cross(USER.PHYS.Velo.unit().normalize()).normalize()
        #     USER.at_normal = WN

        # CAM_ATT_RAD = 0
        # U_PLANE_D = 0
        #

        N_look = USER._pos + Vector3(USER.mat.forward) * -1.0 #camera_distance
        BOX.setClipBounds(USER._pos, N_look, Vector3(USER.mat.up))
        target = target_entity(USER)

        #if target:

        #target = USER.SCHD.test_bounds(USER._pos, Vector3(USER.mat.forward))
        #//targeting based on poly


        if target:
            glCallList(target_acquired(target))
            USER.targeting['origin'] = USER._pos
            USER.targeting['direction'] = (USER._pos - target._pos).normalize()
            USER.targeting['disttoimpact'] = target._pos.get_distance_to(USER._pos)
            USER.targeting['target'] = target
        else:
            USER.targeting = {}


        # if target: glCallList(draw_sector_poly_hilight(obj, target[2]))

        ##################################################################################################
        ##################################################################################################

        light_d = USER._pos + Vector3(USER.mat.up)*4.0 #PHYS.Pos * 1.1

        glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))

        glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))


        #glCallList(draw_utility_line(USER.PHYS.Pos, Vector3(USER.mat.right), Vector3(USER.mat.right)))


        glCallList(obj.gl_list)
        glCallList(star_gl_list)

        #glCallList(draw_subsector_poly_centers())
        #glCallList(BOX.showBounds())

        if TRACE_ENABLE: glCallList(USER.TRAC.draw(USER._pos))


        #
        # debug(['USR  ' + str(USER.message),
        #        'ALT  ' + str(USER.altitude)[0:5],
        #        'STA  ' + str(USER.state_message),
        #        'ADD  ' + str(USER.address),
        #        'CAM  ' + str(CAM.message),
        #        'FPS  ' + str(clock.get_fps())[0:4]])

        if target:
            draw_text(1.0, target.real_name+' '+str(target.spawn_count), target._pos, target.radius)
            if target.flaming:
                mms = str(int((target.radius/target.init_radius)*100))+'%'
                draw_text(2.0, mms, target._pos, target.radius)

        physical_world_element_collisions()

        for BLOB in BLOBS:
            BLOB.update(time_passed_seconds)
            BLOB.move_entity()

            dx = (BLOB._pos-USER._pos)
            de = dx.cross(Vector3(USER.mat.up))
            df = de.cross(Vector3(USER.mat.up))

            dist = str(dx.length)[0:4]+'m'

            draw_text(0.6, dist, USER._pos-df.normalize(), 0.0)

            USER.entity_marker_arrow_angles[BLOB.real_name] = (get_prox(USER.mat, USER._pos, BLOB._pos), dx.length)

            glCallList(BLOB.draw_entity())
            if TRACE_ENABLE: glCallList(BLOB.TRAC.draw(BLOB._pos))

        for fl in Flag.Flags:
            fx = (fl.position - USER._pos)
            glCallList(fl.draw_flag())
            USER.flag_marker_arrow_angles[fl.id] = (get_prox(USER.mat, USER._pos, fl.position), fx.length)

            if fx.length < 0.75:
                snd('affirm-beep', 0.25)

            if fx.length < 0.25:
                del(USER.flag_marker_arrow_angles[fl.id])
                Flag.Flags.remove(fl)
                snd('ah')
            #//NOW NEXT LEVEL

        #print(time_passed_seconds)
        glCallList(FLAM.show())
        glCallList(SMOKES.show())
        glCallList(draw_world_user_line(USER._pos, USER.surface_position))
        glCallList(draw_user())

        USER.PHYS.set(ship_direction_vector)
        USER.update(time_passed_seconds)

        CAM.update_pos(USER._pos)
        glLoadMatrixf(CAM.get_view())

        pygame.display.flip()
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def score(num, target):
    USER.score += num
    Score(num, target)
    pass

def fire_special():
    return fire_special.__name__
    pass

def jump():
    USER.jump()
    snd('jets')
    return jump.__name__
    pass

def pause():
    run_x29.pause_trigger = 1
    return pause.__name__
    pass

def stat():
    run_x29.stat_trigger = 1
    return stat.__name__
    pass

def brakes():
    USER.PHYS.set_deccel()
    return brakes.__name__
    pass

def view():
    snd('beep', 0.75)
    CAM.next_mode()
    return view.__name__
    pass

def shield():
    USER.radius += 0.1
    return shield.__name__
    pass

def fire():
    if USER.targeting:
        FLAM.fire(USER.targeting['origin'],
                  USER.targeting['direction'],
                  USER.targeting['disttoimpact'],
                  USER.targeting['target'])
        snd('fire', 0.6)
    else:
        snd('beep', 0.1)

    return fire.__name__
    pass


def x29_runtime(timedelta, has_input):
    state = 1
    global rotation_direction, movement_direction, ship_direction_vector

    light_d = USER._pos + Vector3(USER.mat.up) * 4.0
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 300, 0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))


    glCallList(WORLD.obj.gl_list)
    glCallList(star_gl_list)
    glCallList(draw_world_user_line(USER._pos, USER.surface_position))

    glCallList(draw_sub_sector_verts())

    if TRACE_ENABLE: glCallList(USER.TRAC.draw(USER._pos))

    # debug(['SCO  ' + str(USER.score),
    #        'USR  ' + str(USER.message),
    #        'ALT  ' + str(USER.altitude)[0:5],
    #        'STA  ' + str(USER.state_message),
    #        'ADD  ' + str(USER.address),
    #        'CAM  ' + str(CAM.message),
    #        'FPS  ' + str(clock.get_fps())[0:4]])

    ship_direction_vector.set(0.0, 0.0, 0.0)

    if has_input:
        rotation = rotation_direction * radians(rotation_speed) #radians(rotation_speed*90.0)
        USER.apply_rotation(rotation)
        heading = Vector3(USER.mat.forward)
        movement = heading.unit() * movement_direction.z * user_movement_speed
        ship_direction_vector += movement  # * time_passed_seconds
        heading_r = Vector3(USER.mat.right)
        movement = heading_r.unit() * movement_direction.x * user_movement_speed
        ship_direction_vector += movement  # * time_passed_second

    USER.PHYS.set(ship_direction_vector)

    USER.update(timedelta)
    CAM.update_pos(USER._pos)
    glLoadMatrixf(CAM.get_view())

    N_look = USER._pos + Vector3(USER.mat.forward) * -1.0  #camera_distance
    BOX.setClipBounds(USER._pos, N_look, Vector3(USER.mat.up))
    target = target_entity(USER)

    if target:
        glCallList(target_acquired(target))
        USER.targeting['origin'] = USER._pos
        USER.targeting['direction'] = (USER._pos - target._pos).normalize()
        USER.targeting['disttoimpact'] = target._pos.get_distance_to(USER._pos)
        USER.targeting['target'] = target
        draw_text(1.0, target.real_name + ' ' + str(target.spawn_count), target._pos, target.radius)
        if target.flaming:
            mms = str(int((target.radius / target.init_radius) * 100)) + '%'
            draw_text(2.0, mms, target._pos, target.radius)
    else:
        USER.targeting = {}

    physical_world_element_collisions()

    for BLOB in NPCs.NPCs_LIST:
        BLOB.update(timedelta)
        BLOB.move_entity()
        dx = (BLOB._pos - USER._pos)
        de = dx.cross(Vector3(USER.mat.up))
        df = de.cross(Vector3(USER.mat.up))
        dist = str(int(floor(dx.length))) + 'm'
        draw_text(0.6, dist, USER._pos - df.normalize(), 0.0)
        USER.entity_marker_arrow_angles[BLOB.real_name] = (get_prox(USER.mat, USER._pos, BLOB._pos), dx.length)
        glCallList(BLOB.draw_entity())
        if TRACE_ENABLE: glCallList(BLOB.TRAC.draw(BLOB._pos))

    for fl in Flag.Flags:
        fx = (fl.position - USER._pos)
        glCallList(fl.draw_flag())
        USER.flag_marker_arrow_angles[fl.id] = (get_prox(USER.mat, USER._pos, fl.position), fx.length)
        if fx.length < 0.75: snd('affirm-beep', 0.25)
        if fx.length < 0.25:
            score(2000,fl.position)
            del (USER.flag_marker_arrow_angles[fl.id])
            Flag.Flags.remove(fl)
            snd('ah')
            #//NOW NEXT LEVEL

    if len(Flag.Flags) == 0: state = 0

    glCallList(FLAM.show())
    glCallList(SMOKES.show())
    glCallList(draw_user())

    Score.show()

    return state
    pass

def x29_level_transition():

    for i in range(0, 100):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_text(0.75, str("LEVEL %d CLEARED!" % current_level), USER._pos, 0.15)
        draw_text(0.5, 'Score: '+str(i*(USER.score/100)), USER._pos, 0.0)
        pygame.display.flip()

    pygame.time.wait(1000)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_text(0.75, str("LEVEL %d BONUS" % current_level) , USER._pos, 0.15)
    bonus = current_level*10000
    USER.score += bonus
    draw_text(0.5, str(bonus)+' points! Score: ' + str(USER.score), USER._pos, 0.0)

    pygame.display.flip()

    pygame.time.wait(2000)




def get_keyboard_input():
    global key_rep_frame, key_rep_max, control_vectors, camera_distance
    no_rep_keys = []
    rep_keys = []
    control_vectors = {'d':[0.0,0.0,0.0],'r':[0.0,0.0,0.0], 'f':[0]}

    def mixin(key,term):
        ts = term.cod.split()
        control_vectors[ts[0]][int(ts[1])] = float(ts[2])
        return key

    for evt in pygame.event.get():
        if evt.type == KEYDOWN:
            #print(evt.key)
            key_rep_frame = 0
            if evt.key == K_q and pygame.key.get_mods() & pygame.KMOD_LMETA:
                quit_x29()
            elif evt.key == K_q and pygame.key.get_mods() & pygame.KMOD_LALT:
                quit_x29()
            elif evt.key == K_ESCAPE:
                quit_x29()
            no_rep_keys = [globals()[k]() for k, v in key_input.items() if evt.key == v.num and 'cod' not in v]

        elif evt.type == KEYUP:
            key_rep_frame = 1
        elif evt.type == QUIT:
            quit_x29()
        elif evt.type == 5:
            if evt.button == 5:
                camera_distance *= 1.1
            elif evt.button == 4:
                camera_distance *= 0.9
        elif evt.type == 4:
            i,j = evt.rel
            control_vectors['r'][0] = j*0.5
            control_vectors['r'][1] = j * 0.5

    #motion keys detached from KEYDOWN/KEYUP
    if key_rep_frame == 0:
        pressed = pygame.key.get_pressed()
        rep_keys = [mixin(k,v) for k, v in key_input.items() if pressed[v.num] and 'cod' in v]

    if key_rep_frame == key_rep_max:
        key_rep_frame = 0
    else:
        key_rep_frame += 1

    return (no_rep_keys,rep_keys)

def run_x29(running):
    StackTrace('run_x29 STARTED')
    run_x29.stat_trigger = 0
    run_x29.pause_trigger = 0
    run_x29.paused = 0
    CAM.set_mode('ground')
    clock.tick()
    global rotation_direction, movement_direction, movement_speed, init_world_size, current_level

    while running:
        has_input = 0
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        time_passed = clock.tick() #(30)
        time_passed_seconds = float(time_passed) / 1000.0

        keys, rep_keys = get_keyboard_input()
        if rep_keys: has_input = 1

        if control_vectors['f'][0] != 0: print(fire())
        cv = control_vectors

        rotation_direction.set(cv['r'][0],cv['r'][1],cv['r'][2])
        movement_direction.set(cv['d'][0],cv['d'][1],cv['d'][2])

        if run_x29.stat_trigger:
            if keys: print(keys)
            if rep_keys: print(control_vectors)
            print('SMOKES',len(SMOKES.smokes))
            print(USER._pos)
            print(CAM._pos)
            print(CAM.mode)
            print(has_input)
            print(rotation_direction)
            print(movement_direction)
            print('stat_trigger')
            run_x29.stat_trigger = 0

        if run_x29.pause_trigger:
            run_x29.paused += 1
            run_x29.pause_trigger = 0

        if run_x29.paused:
            draw_text(1.0, str('Space Program x29 PAUSED'), USER._pos, 0.0)
            if run_x29.paused == 2:
                run_x29.paused = 0
        else:
            state = x29_runtime(time_passed_seconds,has_input)

            if not state:
                # // if not state: change program action here
                x29_level_transition()
                movement_speed += 0.5
                init_world_size += 2
                current_level += 1
                x = x29_world_init(init_world_size)
                clock.tick()


        pygame.display.flip()













if __name__ == "__main__":

    x29_first_run()
    x29_world_init(init_world_size)

    run_x29(True)

    pass
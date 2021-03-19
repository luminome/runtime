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
        wait_time = 100
        dt = time.time()
        self.current_message = str('{:.4f}'.format(dt-self.timer))+" "+str(args)
        print(str('{:.4f}'.format(dt-self.timer))+"\t"+str(args))
        if kwargs:
            if 'no-delay' in kwargs: wait_time = 0
            for earg in kwargs:

                print(str('{:.4f}'.format(dt-self.timer))+"\t"+str(earg))
        print "\n"
        self.StackTraceBlock.append(self.current_message)
        self.timer = dt
        syslog.syslog(syslog.LOG_ALERT, self.current_message)

        if CAM:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_text(1.0, str(self.current_message), USER._pos, 0.0)
            pygame.display.flip()

        pygame.time.wait(wait_time)

    def get_time(self):
        dt = time.time()
        self.current_message = str('{:.4f}'.format(dt - self.blocktime)) + "\tSeconds Elapsed"
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
snd.add("audio/Refuckulate.wav", 'refuck', 0.5)

snd.add("audio/JamesGang.wav", 'james-gang', 0.5)
#snd.add("audio/rediculous.aif", 'redic', 0.4)
snd('amb', 0.4, -1)
snd('amb2', 0.2, -1)
#snd('redic', 0.2, -1)

snd.add('audio/jets_bip.wav', 'jets', 0.5)
snd.add('audio/Wilhelm.wav', 'wilhelm', 0.5)
snd.add('audio/tonethud2.wav', 'thud', 0.75)
snd.add('audio/beep_2.wav', 'beep', 0.5)
snd.add('audio/flame.wav', 'laser', 0.3)
snd.add('audio/small-fire_bip.wav', 'fire', 0.2)
snd.add('audio/02 Nasty Spell_3_bip.wav', 'welcome', 0.6)
snd.add('audio/explode-3.wav', 'explode', 0.85)
snd.add('audio/aaah_bip.aif', 'ah', 1.0)
snd.add('audio/impactexplosion.wav', 'impact', 0.5)
snd.add('audio/shot2.aiff', 'hit', 0.5)
snd.add('audio/ping.aif', 'tweak', 0.5)
snd.add('audio/negative.wav', 'nega', 0.5)
snd.add('audio/confirm.wav', 'confirm', 0.5)
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

class WorldModelSet(object):
    obj = {}
    subobj = {}
    POLY_CENTERS_ARRAY = {}
    POLY_RADII_ARRAY = {}
    SUBSECTOR_ARRAY = {}
    SUBSECTOR_CENTERS_ARRAY = []
    SELECTED_APEX_VERTS = []
    STAR_ARRAY = []
    SUB_SUPER_MEGA = {}
    ready_state = 0
    radius = 8.0
    model_world_max_radius = model_world_radius

    def __init__(self):
        self.mainmodel = "untitled.obj"
        self.msubmodel = 'untitled-sub.obj'
        StackTrace("initialized world model set [class WorldModelSet]")
        pass


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
            vA = (v0 - v1)
            vB = (v0 - v2)
            vC = vA.cross(vB).normalize()
            #vC.normalize()

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
        vts = [1,2,0]
        first_run = True if len(self.POLY_RADII_ARRAY) == 0 else False
        if first_run: self.POLY_CENTERS_ARRAY = {}

        def add_mega(a,b,i):
            if str(a) not in self.SUB_SUPER_MEGA: self.SUB_SUPER_MEGA[str(a)] = {}
            self.SUB_SUPER_MEGA[str(a)][str(b)] = i

        for face in self.obj.faces:
            vertices, normals, texture_coords, material = face
            if first_run: self.POLY_CENTERS_ARRAY[index] = {}
            if first_run: self.POLY_CENTERS_ARRAY[index]['hverts'] = []
            if first_run: self.POLY_CENTERS_ARRAY[index]['nearest'] = []
            v1 = Vector3()
            vs = []
            for ii in range(len(vertices)):
                vg = Vector3((self.obj.vertices[vertices[ii] - 1]))
                vs.append(vg)
                if first_run: self.POLY_CENTERS_ARRAY[index]['hverts'].append(vertices[ii] - 1) #, index))
                if first_run: add_mega(vertices[ii] - 1, vertices[vts[ii]] - 1, index)
                v1 += vg

            v1 /= 3.0
            self.POLY_CENTERS_ARRAY[index]['center'] = v1
            self.POLY_CENTERS_ARRAY[index]['normal'] = getNormal(vs[0], vs[1], vs[2])

            index += 1


        if first_run:
            poly_len = len(self.POLY_CENTERS_ARRAY)
            poly_mod = int(poly_len/10)

            for ii in self.POLY_CENTERS_ARRAY:
                vst = self.POLY_CENTERS_ARRAY[ii]['hverts']
                self.POLY_CENTERS_ARRAY[ii]['nearest'] = list(set([mm for dd in vst for mm in self.SUB_SUPER_MEGA[str(dd)].values()]))

                for jj in range(0,3):
                    vst = list(set(vst+[int(c) for bv in vst for c,i in self.SUB_SUPER_MEGA[str(bv)].items()]))

                self.POLY_RADII_ARRAY[ii] = vst

                if (ii % poly_mod) == 0:
                    StackTrace("configuring radii for poly [%i of %i]" % (ii,poly_len), 'no-delay')
                    snd('system-beep', 0.3)

            snd('affirm-beep',0.5)

        fl = len(self.POLY_CENTERS_ARRAY)
        StackTrace("first_run:"+str(first_run)+" POLY_CENTERS_ARRAY", fl, self.POLY_CENTERS_ARRAY[fl - 1])
        pass


    def build_sector_addresses(self):
        #//THIS SHOULD BE USING POINT IN POLY, NOT INTERSECT-TESTING
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

            self.SUBSECTOR_ARRAY[index] = {'id': str(index) + 'iso',
                                           'center': zip_face([va, vb, vc]),
                                           'sectors': {},
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
                va = ksub['v'][0].get_normalised()
                vb = ksub['v'][1].get_normalised()
                vc = ksub['v'][2].get_normalised()

                Pn = getNormal(va,vb,vc)

                vertset = []
                for kk in POLY_CENTERS_ARRAY_SHADOW:
                    #point_in_poly doesn't work here because it's not height-aware
                    #now it is.
                    #nil = Vector3(0.0, 0.0, 0.0)
                    Pv = self.POLY_CENTERS_ARRAY[kk]['center'].get_normalised()
                    #Pn = self.POLY_CENTERS_ARRAY[kk]['normal']
                    #d = intersect_test(nil, v, va, vb, vc)
                    d1 = point_in_poly(Pv, va, vb, vc)
                    d2 = (Pv-va).dot(Pn)

                    if d1 and (d2 < 0): #d and (d[0] > 0.0):
                        ksub['vertmap'].append(kk)
                        dist = (Pv-ksub['center']).length
                        vertset.append([dist, kk, Pv])
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

            StackTrace('built zones for sector '+self.SUBSECTOR_ARRAY[index]['id'])

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

        snd('refuck')

        self.build_polygon_centers()
        StackTrace("finished building build_polygon_centers")

        # self.build_polygon_radii()
        # StackTrace("finished build_polygon_radii")

        # self.build_sector_addresses()
        # StackTrace("finished building build_sector_addresses")

        self.build_stars(stars_count)
        StackTrace("finished building build_stars")


    def load_world(self, apex_nodes, level):
        self.SELECTED_APEX_VERTS = []
        Upgrade.Upgrades_All = []
        StackTrace("loading world")

        self.build_polygon_topography('nodal_random', apex_nodes, 0.02)
        #build_polygon_topography(obj, 'arbitrary_random', 16, 0.02)
        self.obj.refresh()

        for st in range(0,10):
            r_poly = int(random.randrange(0, len(WORLD.POLY_CENTERS_ARRAY)))

            r_vert = WORLD.POLY_CENTERS_ARRAY[r_poly]['hverts'][0]

            f = random.choice([k for k in upgrades for dummy in range(upgrades[k]['rarity'])])
            f = f if not upgrades[f]['on'] else 'shield'

            Upos = WORLD.POLY_CENTERS_ARRAY[r_poly]['center']
            Unorm = WORLD.POLY_CENTERS_ARRAY[r_poly]['normal']

            upg = Upgrade(f, r_vert, Upos, Unorm)

        StackTrace("finished build_polygon_topography")

        self.build_polygon_centers()

        StackTrace("finished building build_polygon_centers")

        max_heights = []

        flag_ctr = 0
        for vert in self.SELECTED_APEX_VERTS:
            if flag_ctr <= level+1:
                FLAG = Flag()
                FLAG.poly =  vert
                FLAG.position = Vector3(self.obj.vertices[vert])
                max_heights.append(FLAG.position.length)
                flag_ctr += 1

        self.model_world_max_radius = max(max_heights)



        sta = ("finished adding %d flags" % apex_nodes)

        StackTrace(sta)




class NPCSet(object):
    NPCs_LIST = []
    def __init__(self):
        StackTrace("initialized NPC set [class NPCSet]")
        pass

    def generate_blob_entities(self, howmany):
        for en in NPCSet.NPCs_LIST: NPCSet.NPCs_LIST.remove(en)
        for en in PhysicalWorldElement.PWElements:
            if en.name != "user": PhysicalWorldElement.PWElements.remove(en)
        # m = 2.0
        # a = random.random() * m - m / 2.0
        # b = random.random() * m - m / 2.0
        # c = random.random() * m - m / 2.0
        # ev = Vector3(a, b, c)
        """the class of red blobby npcs"""
        for l in range(0, howmany):
            ENTT = EntityHandler('Entity-00' + str(l))

            #for npc in NPCs.NPCs_LIST:
            ENTT.PHYS.set_position(ENTT.ADDR.init_random_poly() * 1.5)

            #ENTT.PHYS.set_position(ev * model_world_radius)
            NPCSet.NPCs_LIST.append(ENTT)

        StackTrace("initialized %d NPCs" % howmany)




class Score_e(dict):
    pass


class PhysicalWorldElement(object):
    """handle everything
    parent class of anything that moves and interacts with material world"""
    #//HANDLE ALL THE THINGS!
    name = ''
    PWElements = []

    def __init__(self, name, radius):
        self.PHYS = WorldPhysics()
        self.TRAC = Trace()
        self.ADDR = PolyAddressHandle(name)
        self.at_poly = 0
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
        amt = by[2]
        self.radius *= amt
        self.mass *= amt
        self.PHYS.Velo += by[0]*-1.0
        self.flaming = 1
        snd('impact',(0.75 / by[1]))
        fires = 40 if upgrades['insta_kill']['on'] else 10

        SMOKES.gimme_a_fucking_smoke(self._pos, fires, 1.0, 'fire')

        if upgrades['insta_kill']['on'] or (self.radius/self.init_radius < 0.1):
            delegated_flames = [f for f in FLAM.flames if f.target.name == self.name]
            for f in delegated_flames: FLAM.flames.remove(f)
            SMOKES.gimme_a_fucking_smoke(self._pos, 20, 5.0)
            snd('explode', (0.85 / by[1]))

            if upgrades['insta_kill']['on']: snd('wilhelm', (1.0 / by[1]))

            self.killed = 1
            score(10000, self._pos)
            self.respawn()
        else:
            score(1000, self._pos)






    def hit(self, sta):
        """when a hit is triggered"""
        if not sta: return
        if not USER.hitting:
            score(100, self._pos)

        USER.hitting = 1
        USER.radius *= 0.95
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
        self.ADDR._pos = self._pos
        self.ADDR._nextpos = self._nextpos
        self.ADDR.idle()

        attitude_deg = 0.0
        a_height = self.ADDR.alt
        a_normal = self.ADDR.nor

        self.at_normal = -a_normal.copy()
        vel = self.PHYS.Velo.length
        ldl = (current_position.length + a_height) / current_position.length
        self.surface_position = ldl * current_position
        self.altitude = a_height*-1.0

        if (self.altitude < 0.2) and not self.PHYS.Jumpy:
            try:
                attitude_deg = degrees(asin(self.PHYS.Velo.get_normalised().dot(a_normal)))
            except ValueError as degree_error:
                self.visible_output_message((attitude_deg, degree_error),timepassed)
                print(self.visible_output_message)

            self.PHYS.Pos = (self.surface_position.length + 0.2) * self.surface_position.get_normalised()

            if (attitude_deg > 45.0) and (vel > 3.0):
                self.PHYS.bounce(a_normal)
                ld = (self._pos-USER._pos).length if self.name != "user" else 1.0
                snd('thud', (0.75 / ld))
                SMOKES.gimme_a_fucking_smoke(self.surface_position, 10, 2.0)
            else:
                self.PHYS.Velo *= surface_velocity_decay
                self.PHYS.creep(a_normal)


    @property
    def _pos(self):
        return self.position.copy()

    @property
    def _nextpos(self):
        return self.PHYS.NextPos.copy()

    pass


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
        self.sub_sector_poly_number = 0
        self.selected_vert_map = []
        self.aux_vert_map = []
        self.seek_vector = Vector3()
        self.seek_vector_normal = Vector3()
        self.message = 'init'

#//NEW PolyAddressHandle REPLACED SCHD:SectorHandle
class PolyAddressHandle(object):
    """referenced as .ADDR"""
    def __init__(self,name):
        self._pos = Vector3()
        self._nextpos = Vector3()
        self.poly_a = Score_e()
        self.poly_b = Score_e()
        self.at_poly = 0
        self.alt = 0
        self.nor = 0
        self.message = 'None'
        self.name = name
        pass

    def localize_poly(self, poly, index):
        rempoly = WORLD.POLY_CENTERS_ARRAY[index]
        poly._id = index
        poly._pos = rempoly['center'].copy()
        poly._ver = rempoly['hverts']
        poly._nea = rempoly['nearest']
        poly._nor = rempoly['normal'].copy()
        poly._set = [Vector3(WORLD.obj.vertices[poly._ver[0]]).normalize(),
                     Vector3(WORLD.obj.vertices[poly._ver[1]]).normalize(),
                     Vector3(WORLD.obj.vertices[poly._ver[2]]).normalize()]
        poly._nat = Vector3(WORLD.obj.vertices[poly._ver[0]])
        return poly

    def init_random_poly(self):
        r = int(random.randrange(len(WORLD.POLY_CENTERS_ARRAY)))
        pp = self.localize_poly(self.poly_a, r)
        self.at_poly = r
        print(self.name, r,"verts:%s near(POLY_CENTERS):%s" % (str(pp._ver), str(pp._nea)))
        return pp._pos.copy()

    def check(self, poly, pos):
        return point_in_poly(pos.get_normalized(), poly._set[0], poly._set[1], poly._set[2])

    def get_new_poly(self, pos):
        vct = 0
        for nearc in self.poly_a._nea:
            polypf = Score_e()
            testee = self.localize_poly(polypf, nearc)
            d = self.check(testee, pos)
            if d: return [vct, nearc, d, testee]
            vct += 1
        return (vct,0,0,0)

    def idle(self):
        self.message = 'None'
        p_current = self.check(self.poly_a, self._pos)

        if not p_current:
            ct, pid, p_current, poly = self.get_new_poly(self._nextpos)
            if pid:
                self.message = ('%i %i > %i' % (ct, self.poly_a._id, pid))
                self.poly_a = poly
            else:
                self.message = 'fuckulated'

        if p_current:
            self.at_poly = self.poly_a._id
            self.alt = (self._pos - self.poly_a._nat).dot(self.poly_a._nor)
            self.nor = self.poly_a._nor


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
                    'fz': Vector3(),
                    'btn': Vector3(),
                    'nc': Vector3()}

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
        self.bottom_offset = 0.01
        self.bottom_surface_pos = Vector3()
        self.poly_confirm = 0

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

    def get_ground_interfere(self):
        for npoly in WORLD.POLY_CENTERS_ARRAY[USER.at_poly]['nearest']:
            pt = WORLD.POLY_CENTERS_ARRAY[npoly]['hverts']
            va = Vector3(WORLD.obj.vertices[pt[0]])
            vb = Vector3(WORLD.obj.vertices[pt[1]])
            vc = Vector3(WORLD.obj.vertices[pt[2]])
            d = intersect_test(world_center, self.fpl['nc'], va, vb, vc)
            if d:
                self.bottom_surface_pos = d[0] * self.fpl['nc']
                return




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
        aux = self.fpl['nc'] + Y * self.nc_h
        if aux.length < self.bottom_surface_pos.length: aux = self.bottom_surface_pos.copy()

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
                                    [0.35, 0.35, 0.35, 1.0]])

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

    flames = []

    def __init__(self):
        self.flames_max = 20
        self.firetimer = 0
        self.fireinterval = 5
        self.firespeed = 0.2
        pass

    def fire(self, shooter_obj, direction_v, target_distance, ptarget):

        if upgrades['rapid_fire']['on']:self.firespeed = 0.8

        flam = Score_e()
        flam.density = 0.5 if upgrades['strong_weapon']['on'] else 0.75
        flam.laser = upgrades['laser']['on']
        flam.origin = shooter_obj._pos
        flam.pos = shooter_obj._pos
        flam.direction = direction_v
        flam.distanceto = target_distance
        flam.target = ptarget
        flam.active = 1
        self.flames.append(flam)
        USER.PHYS.Velo += direction_v*0.2

    def show(self):

        gl_fire_list = glGenLists(2)
        glNewList(gl_fire_list, GL_COMPILE)

        for f in self.flames:
            if f.laser:
                glPushMatrix()
                glColor4f(1.0, 1.0, 0.0, 1.0)
                glBegin(GL_LINES)
                glVertex3f(f.pos.x, f.pos.y, f.pos.z)
                glVertex3f(f.target._pos.x, f.target._pos.y, f.target._pos.z)
                glEnd()
                glPopMatrix()
                f.target.flamed([f.direction, f.distanceto, f.density])
                f.target.hitting = 1
                f.active = 0
            else:
                d = (f.target._pos-f.pos)
                f.pos += (d.get_normalized()*self.firespeed)*0.5
                if (d.length < f.target.radius) and f.active:
                    f.active = 0
                    f.target.flamed([f.direction,f.distanceto,f.density])
                    f.target.hitting = 1
                else:
                    if (f.pos.length < f.target._pos.length) and (f.target._pos.length <= f.origin.length): f.pos = f.target._pos.length * f.pos.get_normalized()
                    glPushMatrix()
                    glTranslate(f.pos.x, f.pos.y, f.pos.z)
                    glColor4f(1.0, 1.0, 0.0, 1.0)
                    glutSolidSphere(0.035,10,10)
                    glPopMatrix()

        glEndList()

        f_kill = [Flame.flames.remove(f) for f in Flame.flames if not f.active]

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


class EntityHandler(PhysicalWorldElement):
    """instance class for entities"""
    #//Entity Handler
    def __init__(self, name):
        er = random.random() * 0.5 + 0.5
        super(EntityHandler, self).__init__(name, er)
        aa = int(random.random() * len(names_one))-1
        bb = int(random.random() * len(names_one))-1
        self.real_name = str(names_one[aa] + '-' + names_one[bb])
        names_one.remove(names_one[aa])
        names_one.remove(names_one[bb])
        self.mass = er * 10.0
        self.apply_rotation_to_mat(rotation_direction)
        self.behavior_idle = random.random()*180.0
        self.rotation_amt = 0
        self.spawn_count = 0
        self.PHYS.Velo = Vector3(0.001,0.001,0.001)
        self.respawn_timer = 0
        self.active = 1


    def respawn(self):
        m = 2.0
        er = random.random() * 0.5 + 0.5
        # a = random.random() * m - m / 2.0
        # b = random.random() * m - m / 2.0
        # c = random.random() * m - m / 2.0
        # ev = Vector3(a, b, c)

        self.radius = er
        self.init_radius = er
        self.PHYS.Velo = Vector3(0.0001, 0.0001, 0.0001)

        #self.PHYS.set_position(ev * (WORLD.model_world_max_radius))

        self.killed = 0
        self.spawn_count += 1
        self.respawn_timer = ceil(er*1000)
        self.active = 0
        if self.real_name in USER.entity_marker_arrow_angles: del USER.entity_marker_arrow_angles[self.real_name]


    def hold_idle(self, timepassed):
        self.PHYS.update(timepassed)
        self.respawn_timer -= 1
        self.active = self.respawn_timer < 0
        if self.active: self.PHYS.set_position(self.ADDR.init_random_poly() * 1.5)


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
        self.radar = 1

    def jump(self, height):
        self.PHYS.Jumpy = 1
        self.PHYS.Velo += self.at_normal * height

    def apply_rotation(self, rotation):
        if CAM.mode == 'ground':
            self.ground_offset_height += rotation[0]
            if self.ground_offset_height < 0: self.ground_offset_height = 0.0
            rotation[0] *= 0.0
        super(UserHandler, self).apply_rotation_to_mat(rotation)


class Flag(object):
    Flags = []
    def __init__(self):
        self.id = ('flag 0%i' % (len(self.Flags)+1) )
        self.position = Vector3()
        self.animation_vector = Vector3()
        self.animation_frames = 60.0
        self.animation_ct = 0.0
        self.flag_graphic = flag#flag_d()
        self.state = "open"
        self.poly = 0
        self.Flags.append(self)

    def draw_flag(self):
        if self.state != "open":
            self.animation_ct += 1.0
            self.animation_vector = self.position.get_normalized() * 0.01 * self.animation_ct

        if self.animation_ct > self.animation_frames: self.close()

        self.position += self.animation_vector

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

    def close(self):
        Flag.Flags.remove(self)

    def acquire(self):
        if self.state == 'open':
            if self.id in USER.flag_marker_arrow_angles: del USER.flag_marker_arrow_angles[self.id]
            self.state = 'acquired'
            score(2000, self.position)
            snd('ah')


class ScoreMarker(object):
    scores = []

    def __init__(self):
        pass

    def __call__(self, points, pos):
        s = Score_e()
        s.pos = pos
        s.points = points
        s.frames = 38
        self.scores.append(s)
        pass

    def show(self):
        for s in self.scores:
            s.frames -= 1
            s.pos *= 1.001
            draw_text(1.0, str(s.points), s.pos, 0.0)
        r = [ScoreMarker.scores.remove(s) for s in ScoreMarker.scores if s.frames < 0]

        scpos = CAM._pos + Vector3(CAM.rotmat.forward)*-2.0
        draw_text(0.5, str(USER.score), scpos, 0.4)

    pass


class Upgrade(object):
    #//upgrades per sector
    Upgrades_All = []

    def define_mat_from_normal(self):
        PPN = self.position.get_normalized()
        UF = self.normal.normalize()
        PN = UF.cross(PPN).normalize()
        PW = PN.cross(PPN).normalize()
        self.mat.set_row(0, PN.as_tuple())
        self.mat.set_row(1, PPN.as_tuple())
        self.mat.set_row(2, PW.as_tuple())

    def __init__(self, upgradetype, subpoly, poly_center, poly_normal):
        self.start_height = 1.01
        self.poly = subpoly
        self.normal = Vector3(poly_normal)
        self.position = Vector3(poly_center)
        self.upgrade_type = upgradetype
        self.upgrade_name = upgradetype.upper().replace('_',' ')
        self.mat = Matrix44()
        self.mat.set_row(3,(self.position*self.start_height).as_tuple())
        self.define_mat_from_normal()
        self.intercepted = 0.0
        self.intercept_frames = 12.0
        self.Upgrades_All.append(self)

    def show(self):

        if self.intercepted > 0:
            dscale = (1.0 - self.intercepted/self.intercept_frames)
            self.intercepted += 1.0
        else:
            dscale = 1.0

        if self.intercepted > self.intercept_frames:
            self.Upgrades_All.remove(self)

        upgrade_gl_list = glGenLists(1)
        glNewList(upgrade_gl_list, GL_COMPILE)
        glPushMatrix()
        glMultMatrixf(self.mat.to_opengl())
        glScale(dscale, dscale, dscale)
        glCallList(upgrade)
        glPopMatrix()
        glEndList()
        return upgrade_gl_list
        pass

    def intercept(self):
        if self.intercepted == 0.0:
            snd('confirm')

            if self.upgrade_type == 'shield':
                USER.radius += 0.25
            else:
                upgrades[self.upgrade_type]['on'] = True

            self.intercepted = 1


class Button(object):


    def __init__(self):


        pass



    pass

#//separated runtime functions
def getNormal(v1, v2, v3):
   a = v1 - v2
   b = v1 - v3
   return a.cross(b).normalize()

def point_in_poly(P, vA, vB, vC):
    av0 = vC - vA
    av1 = vB - vA
    av2 = P - vA
    dot00 = av0.dot(av0)
    dot01 = av0.dot(av1)
    dot02 = av0.dot(av2)
    dot11 = av1.dot(av1)
    dot12 = av1.dot(av2)
    #Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    au = (dot11 * dot02 - dot01 * dot12) * inv_denom
    av = (dot00 * dot12 - dot01 * dot02) * inv_denom
    #Check if point is in triangle
    return (au >= 0.0) and (av >= 0.0) and (au + av < 1.0)

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
        element_collision = [el for el in PhysicalWorldElement.PWElements if (el.name != sel.name) and ((el._nextpos-sel._nextpos).length < (sel.radius + el.radius))]
        for el in element_collision:
            if (el,sel) not in res:
                el.collided = 1 if sel.name == 'user' else 0
                sel.collided = 1 if el.name == 'user' else 0
                el.hit(el.collided)
                sel.hit(sel.collided)
                collide(sel,el)
                res.append((sel,el))

def target_entity(user_obj):
    targeted = [((ent._pos-user_obj._pos).length, ent) for ent in PhysicalWorldElement.PWElements if ent.name != user_obj.name and ent.active and BOX.testBounds(ent._pos)]
    return min(targeted)[1] if len(targeted) else 0

def score(num, target):
    USER.score += num
    Score(num, target)
    pass




#//drawing functions
def draw_user():
    pos = USER._pos
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_ship_scale, user_ship_scale, user_ship_scale)
    glCallList(pyramid)
    glPopMatrix()

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glTranslate(0.0, 0.0, -1.0)
    glScale(user_arrow_scale, user_arrow_scale, user_arrow_scale)
    glCallList(arrow)
    glPopMatrix()

    if USER.radar:
        for aname, angle in USER.entity_marker_arrow_angles.items():
            glColor4f(0.2, 0.7, 0.2, 1.0)
            siz = 0.025 + 0.05 * (2.0/angle[1])
            glPushMatrix()
            glTranslate(pos.x, pos.y, pos.z)
            glMultMatrixf(USER.mat.to_opengl())
            glRotate(degrees(angle[0]), 0.0, 1.0, 0.0)
            glTranslate(0.0, 0.0, -2.0 + (0.5/angle[1]))
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
            glScale(0.2,0.2,0.2)
            glCallList(flag)
            glPopMatrix()

    glFrontFace(GL_CCW)
    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
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

def draw_poly_hilight_group(index):
    t = 1.001
    #poly = USER.SCHD.sub_sector_poly_number
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    glPointSize(4.0)
    glColor3f(1.0, 0.3, 1.0)
    glBegin(GL_POINTS)
    for npoly in WORLD.POLY_CENTERS_ARRAY[index]['nearest']:
        v1 = WORLD.POLY_CENTERS_ARRAY[npoly]['center']
        glVertex3f(v1.x, v1.y, v1.z)
        # if BOX.testBounds(v1):
        #     glVertex3f(v1.x, v1.y, v1.z)
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

def draw_sector_poly_hilight(index):
    """used to indicate the current sector"""
    gl_list = glGenLists(2)
    glNewList(gl_list, GL_COMPILE)
    ph = WORLD.POLY_CENTERS_ARRAY[index]['hverts']
    glColor3f(0.2, 1.0, 0.2)
    a = WORLD.obj.vertices[ph[0]]
    b = WORLD.obj.vertices[ph[1]]
    c = WORLD.obj.vertices[ph[2]]
    glBegin(GL_LINES)
    glVertex3fv(a)
    glVertex3fv(b)
    glVertex3fv(b)
    glVertex3fv(c)
    glVertex3fv(c)
    glVertex3fv(a)
    glEnd()
    glEndList()
        
    return gl_list

def draw_sub_sector_verts(index):
    gl_sub_list = glGenLists(2)
    glNewList(gl_sub_list, GL_COMPILE)
    #vert_map = WORLD.SUBSECTOR_ARRAY[USER.SCHD.sector_number]['sectors'][USER.SCHD.sub_sector_number]['hverts']
    vert_map = WORLD.POLY_RADII_ARRAY[index]

    glPointSize(2.0)
    glBegin(GL_POINTS)
    glColor3f(0.2, 1.0, 0.2)
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

def draw_utility_line(origin,a,color):
    a_gl_list = glGenLists(1)
    glNewList(a_gl_list, GL_COMPILE)
    a = a.copy() - origin
    glPushMatrix()
    glTranslate(origin.x, origin.y, origin.z)
    glColor3f(color[0],color[1],color[2])

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


def arrow_d():
    arrow = glGenLists(1)
    glNewList(arrow, GL_COMPILE)
    glColor4f(0.0, 0.55, 0.0, 0.5)
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

    glColor4f(0.1, 0.55, 0.1, 1.0)
    h = float(2.28824561127089)
    s = float(1.41421356237309)

    glBegin(GL_POLYGON)
    a = Vector3(0.0, 0.0, s)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(s, 0.0, 0.0)
    N = getNormal(a,b,c)*0.25
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(-s, 0.0, 0.0)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(0.0, 0.0, s)
    N = getNormal(a, b, c)*0.25
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(0.0, 0.0, -s)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(-s, 0.0, 0.0)
    N = getNormal(a, b, c)*0.25
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(s, 0.0, 0.0)
    b = Vector3(0.0, h, 0.0)
    c = Vector3(0.0, 0.0, -s)
    N = getNormal(a, b, c)*0.25
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
def bar_d():
    bar = glGenLists(1)
    glNewList(bar, GL_COMPILE)
    glFrontFace(GL_CW)
    glColor4f(0.6, 0.4, 0.0, 1.0)

    glBegin(GL_POLYGON)
    a = Vector3(0.1, 0.3, -0.1)
    b = Vector3(0.1, -0.3, -0.1)
    c = Vector3(-0.1, -0.3, -0.1)
    d = Vector3(-0.1, 0.3, -0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(-0.1, 0.3, 0.1)
    b = Vector3(-0.1, -0.3, 0.1)
    c = Vector3(0.1, -0.3, 0.1)
    d = Vector3(0.1, 0.3, 0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(-0.1, 0.3, 0.1)
    b = Vector3(-0.1, 0.3, -0.1)
    c = Vector3(-0.1, -0.3, -0.1)
    d = Vector3(-0.1, -0.3, 0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glBegin(GL_POLYGON)
    a = Vector3(0.1, -0.3, 0.1)
    b = Vector3(0.1, -0.3, -0.1)
    c = Vector3(0.1, 0.3, -0.1)
    d = Vector3(0.1, 0.3, 0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #top
    glBegin(GL_POLYGON)
    a = Vector3(0.1, 0.3, 0.1)
    b = Vector3(0.1, 0.3, -0.1)
    c = Vector3(-0.1, 0.3, -0.1)
    d = Vector3(-0.1, 0.3, 0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #bottom
    glBegin(GL_POLYGON)
    a = Vector3(-0.1, -0.3, 0.1)
    b = Vector3(-0.1, -0.3, -0.1)
    c = Vector3(0.1, -0.3, -0.1)
    d = Vector3(0.1, -0.3, 0.1)
    N = getNormal(a, b, c)
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glEndList()
    return bar
bar = bar_d() #a visual bar element for upgrade ICON (white, vertical)
def upgrade_d():
    upgrade = glGenLists(1)
    glNewList(upgrade, GL_COMPILE)
    glColor4f(0.5, 0.5, 0.5, 1.0)
    glCallList(bar)
    glRotate(90.0, 0.0, 0.0, 1.0)
    glCallList(bar)
    glRotate(90.0, 1.0, 0.0, 0.0)
    glCallList(bar)
    glEndList()
    return upgrade
upgrade = upgrade_d() #upgrade ICON
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
flag = flag_d() #flag ICON


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

### M O D E L V I E W ###
glMatrixMode(GL_MODELVIEW)
### M O D E L V I E W ###

star_gl_list = []

sectors = []
user_message = 'WELCOME'
world_center = Vector3(0.0,0.0,0.0)

time_passed = clock.tick()

WORLD = WorldModelSet()
NPCs = NPCSet()

SMOKES = Corey_Has_Smokes()
FLAM = Flame()

CAM = CameraHandler(camera_distance)
USER = UserHandler('user')
uv = Vector3(1.0,1.0,1.0)


#USER.PHYS.set_position(uv*model_world_radius)




ship_direction_vector = Vector3()
rotation_direction = Vector3()
movement_direction = Vector3()
COR = 1.0
attitude_deg = 0

N = camera_distance*0.25
F = camera_distance*8
BOX = ViewClipBox(N, F, float(width)/(F), float(width)/(F), 0.1)

ship_direction_vector.set(0.0, 0.0, 0.0)

Score = ScoreMarker()

key_rep_frame = 0
control_vectors = {}
stat_trigger = 0

init_world_size = 2
current_level = 1

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||



def fire_special():
    return fire_special.__name__
    pass

def jump():
    ht = 2.0 if upgrades['high_jump']['on'] == False else 8.0
    USER.jump(ht)
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
    snd('jets')
    return brakes.__name__
    pass

def view():
    snd('beep', 0.75)
    CAM.next_mode()
    return view.__name__
    pass

def radar_toggle():
    USER.radar = abs(USER.radar-1)
    return radar_toggle.__name__
    pass

def fire():
    if upgrades['rapid_fire']['on']:
        if USER.targeting:
            FLAM.fire(USER,
                      USER.targeting[1],
                      USER.targeting[2],
                      USER.targeting[3])

            if upgrades['laser']['on']:
                snd('laser', 0.6)
            else:
                snd('fire', 0.6)
        else:
            snd('beep', 0.1)

    return fire.__name__
    pass

def fire_aux():
    if USER.targeting:
        FLAM.fire(USER,
                  USER.targeting[1],
                  USER.targeting[2],
                  USER.targeting[3])
        if upgrades['laser']['on']:
            snd('laser', 0.6)
        else:
            snd('fire', 0.6)
    else:
        snd('beep', 0.1)

    return fire_aux.__name__
    pass

#//separated runtime functions
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



def x29_world_init():
    global movement_speed, init_world_size, current_level
    """Setup the game level"""
    USER.entity_marker_arrow_angles = {}
    USER.flag_marker_arrow_angles = {}

    USER.PHYS.set_position(USER.ADDR.init_random_poly() * 1.5)
    #USER.update
    #USER.ADDR.idle()

    WORLD.load_world(init_world_size, current_level)
    StackTrace('WORLD.load_world(%s) COMPLETE' % init_world_size)

    #PhysicalWorldElement.PWElements = [x for x in PhysicalWorldElement.PWElements if x.name == 'user']

    #//HANDLING OF NPCs IS OFF!!!
    # e = min([int(current_level),4])
    NPCs.generate_blob_entities(1)

    for npc in NPCs.NPCs_LIST:
        npc.PHYS.set_position(npc.ADDR.init_random_poly() * 1.5)


    StackTrace('NPCs.generate_blob_entities COMPLETE')
    StackTrace(len(WORLD.POLY_CENTERS_ARRAY))
    StackTrace.get_time()

    movement_speed += 0.1
    init_world_size += 3
    current_level += 1




    return 'ok'

def quit_x29():
    StackTrace("quitting")
    pygame.quit()
    exit()

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def x29_runtime(timedelta, has_input):
    state = 1
    global rotation_direction, movement_direction, ship_direction_vector

    #m = USER.ADDR.idle()

    # if USER.ADDR.message is not 'None': print(str(USER.ADDR.message))

    light_d = USER._pos + Vector3(USER.mat.up) * 4.0
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 300, 0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))

    glCallList(WORLD.obj.gl_list)
    glCallList(star_gl_list)
    glCallList(draw_world_user_line(USER._pos, USER.surface_position))

    #user_poly = USER.at_poly


    glCallList(draw_poly_hilight_group(USER.ADDR.at_poly))

    #user_poly = USER.SCHD.sub_sector_poly_numbera
    #glCallList(draw_sector_poly_hilight(BOX.poly_confirm))
    #glCallList(draw_sector_poly_hilight())

    glCallList(draw_sub_sector_verts(USER.ADDR.at_poly))

    #glCallList(BOX.showBounds())
    #glCallList(draw_utility_line(world_center, BOX.bottom_surface_pos, (1.0, 1.0, 0.0)))

    if TRACE_ENABLE: glCallList(USER.TRAC.draw(USER._pos))

    debug(['FPS  ' + str(clock.get_fps())[0:4]]) #,
    #       'ALT  ' + str(USER.ADDR.alt)])
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


    if len(NPCs.NPCs_LIST) > 0: physical_world_element_collisions()

    #//NPCs LIST
    for BLOB in NPCs.NPCs_LIST:
        if BLOB.active:
            #if BLOB.ADDR.message is not 'None': print(BLOB.real_name,str(BLOB.ADDR.message))

            BLOB.update(timedelta)
            BLOB.move_entity()

            dx = (BLOB._pos - USER._pos)
            de = dx.cross(Vector3(USER.mat.up))
            df = de.cross(Vector3(USER.mat.up))
            dist = str(int(floor(dx.length))) + 'm'

            if upgrades['enemy_radar']['on'] and USER.radar:
                draw_text(0.6, dist, USER._pos - (df.normalize()*2.0), 0.0)
                USER.entity_marker_arrow_angles[BLOB.real_name] = (get_prox(USER.mat, USER._pos, BLOB._pos), dx.length)

            glCallList(BLOB.draw_entity())
            if TRACE_ENABLE: glCallList(BLOB.TRAC.draw(BLOB._pos))
        else:
            BLOB.hold_idle(timedelta)

    #//FLAGS LIST
    for fl in Flag.Flags:
        glCallList(fl.draw_flag())
        fx = (fl.position - USER._pos)
        if upgrades['flag_radar']['on'] and USER.radar and fl.state == "open": USER.flag_marker_arrow_angles[fl.id] = (get_prox(USER.mat, USER._pos, fl.position), fx.length)

        if fl.poly in WORLD.POLY_RADII_ARRAY[USER.ADDR.at_poly]:
            thr = USER._pos.length > fl.position.length
            if fl.state == "open" and upgrades['flag_magnet']['on'] and (fx.length < 16.0) and thr:
                glCallList(draw_utility_line(USER._pos,fl.position,(1.0,1.0,0.0)))
            if upgrades['flag_magnet']['on'] and (fx.length < 6.0) and thr:
                fl.acquire()
            elif fx.length < 0.5:
                fl.acquire()

    #//UPGRADES LIST
    ups = [u for u in Upgrade.Upgrades_All if u.poly in WORLD.POLY_RADII_ARRAY[USER.ADDR.at_poly]]
    for upg in ups:
        glCallList(upg.show())
        ux = (upg.position - USER._pos)
        if ux.length < 8.0:
            draw_text(0.8, upg.upgrade_name, upg.position, 0.25)
            if ux.length < USER.radius: upg.intercept()

    #//targeting
    N_look = USER._pos + Vector3(USER.mat.forward) * -1.0  #camera_distance
    BOX.get_ground_interfere()
    BOX.setClipBounds(USER._pos, N_look, Vector3(USER.mat.up))
    target = target_entity(USER)

    if target:
        glCallList(target_acquired(target))
        USER.targeting = [None,
                          (USER._pos - target._pos).normalize(),
                          (target._pos-USER._pos).length,
                          target]

        draw_text(1.0, target.real_name + ' ' + str(target.spawn_count), target._pos, target.radius)
        if target.flaming:
            mms = str(int((target.radius / target.init_radius) * 100)) + '%'
            draw_text(2.0, mms, target._pos, target.radius)
    else:
        USER.targeting = []

    #//level flip
    if len(Flag.Flags) == 0: state = 0

    glCallList(FLAM.show())
    glCallList(SMOKES.show())
    glCallList(draw_user())

    Score.show()

    return state
    pass

def x29_level_transition():
    snd('james-gang',0.45)
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
    draw_text(0.5, str(bonus)+' points! Total score: ' + str(USER.score), USER._pos, 0.0)

    pygame.display.flip()

    pygame.time.wait(2000)

    return 1

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

        if 'fire_aux' not in keys and control_vectors['f'][0] != 0:
            fire()
        elif 'fire_aux' in keys:
            fire_aux()

        cv = control_vectors
        rotation_direction.set(cv['r'][0],cv['r'][1],cv['r'][2])
        movement_direction.set(cv['d'][0],cv['d'][1],cv['d'][2])

        if run_x29.stat_trigger:
            if keys: print(keys)
            if rep_keys: print(control_vectors)
            print('UPGRADES', len(Upgrade.Upgrades_All))
            print('SMOKES',len(SMOKES.smokes))
            print('SCORES', len(Score.scores))
            print('FLAMES', len(FLAM.flames))
            print('FLAGS', len(Flag.Flags))
            print('NPCs',len(NPCs.NPCs_LIST))
            print('ELEMENTS',len(PhysicalWorldElement.PWElements))
            print(CAM.mode)
            print('FPS', str(clock.get_fps())[0:4])

            e = [(l,u['on']) for l,u in upgrades.items()]
            print(e)
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
                st = x29_level_transition()
                st = x29_world_init()
                clock.tick()
                return "stop"

        pygame.display.flip()



if __name__ == "__main__":
    x29_first_run()
    x29_world_init()
    def rungame():
        return run_x29(True)

    c = rungame()
    if c == "stop":
        StackTrace("ok")
        rungame()

    pass
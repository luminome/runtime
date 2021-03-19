#! /usr/bin/python
import random
import plistlib
import time,gc,inspect
import syslog
import weakref



import pygame
from pygame.constants import *

# import pyaudio
#
# from pydub import AudioSegment
# from pydub.playback import play

from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL import GLUT
import numpy
from freetype import *
from gameobjects.matrix44 import *
from gameobjects.vector3 import *
from objloader import *


#pygame.mixer.pre_init(44100, -16, 2, 2048*16)
pygame.init()
#pygame.mixer.init(44100,-16,2,1024*4)
pygame.display.set_caption('Space Program x29')

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
        wait_time = 10
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
            snd('ticker-2')
            RENDER_POS = CAM._pos - Vector3(CAM.mat.forward)*float(screen_text_depth)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            #ABS_POS = Vector3(0.0, 0.0, 0.0)
            TEXTB.draw(1.0, self.current_message, RENDER_POS, 0.0)
            TEXTB.draw(0.5, "get game keyboard instructions by pressing ? key after load", RENDER_POS, -0.25)
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

kord = 'a,d,w,s,arrow-left,arrow-right,arrow-up,arrow-down,q,e,space,shift,c,x,v,-,=,p,?'.split(',')
group = [(k,key) for k in kord for key in key_input if key_input[key]['ks'] == k and not 'hidden' in key_input[key].keys()]
globals()['instructions'] = "(key): action\n"
for tupl in group:
    #"%s:\t%s\n"
    kstr = ("( %s ): %s\n" % (tupl[0], tupl[1].replace('_',' ').replace('-',' '))).upper()
    instructions += kstr
instructions += "(ESC)(CMD-Q)(CTRL-Q): QUIT\n"

class BetterTypeHandler(object):
    base, texid = 0, 0
    widths = []
    sp = 0.0
    ke = 0.0
    lh = 0.0

    def __init__(self):
        self.makefont('pf_tempesta_seven.ttf', 16)

    def draw(self, sca, text, pos, ver, *args):
        text = str(text)
        if args and args[0] == 'left':
            xt = 0.0
        else:
            xt = sum([self.widths[ord(c) - 32][0]+self.ke for c in text])
        scale = 0.02*sca
        """line height (lh) is 24...these are for unscaled values"""
        height_offset = ver*1.8
        glPushMatrix()
        glEnable(GL_TEXTURE_2D)
        glDepthMask(GL_FALSE)
        glFrontFace(GL_CW)
        glTranslatef(float(pos.x), float(pos.y), float(pos.z))
        glPushMatrix()
        glMultMatrixf(CAM.mat.to_opengl())
        glColor4f(0.1, 0.9, 0.1, 0.85)
        #glRotatef(180.0, 0.0, 1.0, 0.0)
        glTranslatef(0.0, height_offset, 0.0)
        glScalef(scale, scale, scale)
        glTranslatef((xt) * -0.5, self.lh, 0.0)
        glPushMatrix()
        glListBase(self.__class__.base + 1)
        glCallLists([ord(c) for c in text])
        glPopMatrix()
        glPopMatrix()
        glDepthMask(GL_TRUE)
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

    def makefont(self, filename, size):
        #global texid,widths
        face = Face(filename)
        face.set_char_size(size * 64)

        # Determine largest glyph size
        width, height, ascender, descender = 0, 0, 0, 0
        for c in range(32, 128):
            face.load_char(chr(c), FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT)
            bitmap = face.glyph.bitmap
            width = max(width, bitmap.width)
            ascender = max(ascender, face.glyph.bitmap_top)
            descender = max(descender, (bitmap.rows - face.glyph.bitmap_top))
        height = ascender + descender
        #print(width, height)

        self.lh = float(height)
        self.sp = width * float(0.5)
        self.ke = width * float(0.1)

        # Generate texture data
        Z = numpy.zeros((height * 6, width * 16), dtype=numpy.ubyte)
        for j in range(6):
            for i in range(16):
                face.load_char(chr(32 + j * 16 + i), FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT)

                wg = bitmap.width if bitmap.width != 0 else self.sp

                self.widths.append((wg, face.glyph.bitmap_left))
                x = i * width + face.glyph.bitmap_left
                y = j * height + ascender - face.glyph.bitmap_top
                Z[y:y + bitmap.rows, x:x + bitmap.width].flat = bitmap.buffer

        # Bound texture
        self.texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, Z.shape[1], Z.shape[0], 0, GL_ALPHA, GL_UNSIGNED_BYTE, Z)

        # Generate display lists
        dx, dy = width / float(Z.shape[1]), height / float(Z.shape[0])
        base = glGenLists(8 * 16)

        for si in range(8 * 16):
            c = chr(si)
            x = si % 16
            y = si // 16 - 2
            ww = self.widths[x + y * 16]
            w = ww[0]
            o = ww[1] + 1 if ww[1] else 0
            #if w == 0: w = self.sp
            a = o / float(width)
            z = w / float(width)
            #print(base + si)
            glNewList(base + si, GL_COMPILE)
            if (c == '\n'):
                glPopMatrix()
                glTranslatef(0, -self.lh, 0)
                glPushMatrix()
            elif (c == '\t'):
                glTranslatef(4 * width, 0, 0)
            elif (si >= 32):
                glBegin(GL_QUADS)
                glTexCoord2f((x + a) * dx, (y + 1) * dy), glVertex(0, -height)
                glTexCoord2f((x + a) * dx, (y) * dy), glVertex(0, 0)
                glTexCoord2f((x + z) * dx, (y) * dy), glVertex(w, 0)
                glTexCoord2f((x + z) * dx, (y + 1) * dy), glVertex(w, -height)
                glEnd()
                glTranslatef(w + self.ke, 0, 0)
            glEndList()




class audio(object):
    audio_lex = {}
    repeat = 0
    distance_threshold = 20.0

    def __call__(self, asnd, *args):
        repeat = 0
        """need audio position calculations"""
        if len(args) > 0:
            psnd = self.audio_lex[asnd]
            if args[0].__class__ == Vector3:
                LR = self.mapsoundspace(args[0])
                if LR == None: return
                psnd['sound'].stop()
                channel = psnd['sound'].play()
                channel.set_volume(psnd['basevolume']*LR[1], psnd['basevolume']*LR[0])
                return
            elif args[0] == 'stop':
                psnd['sound'].stop()
                return
            psnd['sound'].set_volume(args[0]*psnd['basevolume'])
            repeat = args[1] if len(args) == 2 else 0

        self.audio_lex[asnd]['sound'].stop()
        self.audio_lex[asnd]['sound'].play(repeat)


    def mapsoundspace(self, snd_pos):
        snd_d_t = self.__class__.distance_threshold
        ub = USER._pos
        snd_dist = (ub - snd_pos).length
        if snd_dist > snd_d_t: return None
        ua = Vector3(USER.mat.forward)
        uc = Vector3(USER.mat.up)
        N = getNormal(ua,ub,uc)
        snd_lr_offset = snd_pos.dot(N)
        base_amount = (snd_d_t - snd_dist) / snd_d_t
        raw = [(snd_d_t - snd_lr_offset) / snd_d_t,(snd_d_t + snd_lr_offset) / snd_d_t]
        norm = [float(i) / sum(raw) for i in raw]
        LR = (2*norm[0]*base_amount,2*norm[1]*base_amount)

        return LR
        pass


    def add(self, resource, snd_name, volume):
        #//pyaudio and pydub import AudioSegment
        res = {'basevolume':volume,
               'sound':pygame.mixer.Sound(os.path.join(BASEDIR, str(resource)))}
        res['sound'].set_volume(volume)
        self.audio_lex[snd_name] = res

snd = audio()


#BACKGROUND SOUNDS
#snd.add("audio/ambient.wav", 'amb', 0.4)
snd.add("audio/ambient2.wav", 'amb2', 0.3)
snd.add("audio/Refuckulate.wav", 'refuck', 0.5)
snd.add("audio/JamesGang.wav", 'james-gang', 0.5)
snd.add("audio/rediculous-mixup.aif", 'redic', 0.3)
#snd('amb', 0.4, -1)
snd('amb2', 0.3, -1)
snd('redic', 0.3, -1)

snd.add('audio/CENA-CLIP_1.wav', 'cena', 0.5)
snd.add('audio/jets_bip.wav', 'jets', 0.5)
snd.add('audio/Wilhelm.wav', 'wilhelm', 0.5)
snd.add('audio/tonethud2.wav', 'thud', 0.75)
snd.add('audio/beep_2.wav', 'beep', 0.5)
snd.add('audio/flame.wav', 'laser', 0.3)
snd.add('audio/flame-basic-user.aif', 'fire', 0.2)
snd.add('audio/small-fire_bip.wav', 'enemy-fire', 0.4)
snd.add('audio/02 Nasty Spell_3_bip.wav', 'welcome', 0.6)
snd.add('audio/explode-3.wav', 'explode', 0.85)
snd.add('audio/aaah_bip.aif', 'ah', 0.6)
snd.add('audio/impactexplosion.wav', 'impact', 0.5)
snd.add('audio/shot2.aiff', 'hit', 0.5)
snd.add('audio/ping.aif', 'tweak', 0.5)
snd.add('audio/ding.aiff', 'bonus', 0.25)
snd.add('audio/negative.wav', 'nega', 0.2)
snd.add('audio/confirm.wav', 'confirm', 0.4)
snd.add('audio/computerbeep_17_bip_1.aif', 'system-beep', 0.15)
snd.add('audio/computerbeep_56.aif', 'affirm-beep', 0.5)
snd.add('audio/nasty-beep.aif', 'nasty-beep', 0.75)
snd.add('audio/red-alert.aif', 'code-red', 0.85)
snd.add('audio/code-alert.aif', 'code-alert', 0.25)
snd.add('audio/cheese.aif', 'cheese', 0.75)
snd.add('audio/all-bravo.aif', 'bravo', 0.5)
snd.add('audio/micro-beep-4.aif', 'ticker', 0.1)
snd.add('audio/micro-beep.aif', 'ticker-2', 0.5)
snd.add('audio/morphin.aif', 'morphin', 0.5)


# aud = BetterAudio()
# aud('audio/rediculous-mixup.aif')





clock = pygame.time.Clock()
viewport = (1280, 800)
hx = viewport[0]/2
hy = viewport[1]/2
#srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
srf = pygame.display.set_mode(viewport, FULLSCREEN | OPENGL | DOUBLEBUF | HWSURFACE | NOFRAME, 32)

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION,  (0, 300, 0, 1.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.48, 0.48, 0.48, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

glEnable(GL_LIGHT1)
glLightfv(GL_LIGHT1, GL_POSITION,  (0, 0, 0, 1.0))
glLightfv(GL_LIGHT1, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 1.0, 0.5, 1.0))
glLightfv(GL_LIGHT1, GL_SPECULAR, (0.5, 1.0, 0.5, 1.0))
glLightfv(GL_LIGHT1, GL_CONSTANT_ATTENUATION, (1.0))
glLightfv(GL_LIGHT1, GL_LINEAR_ATTENUATION, (0.1))
glLightfv(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, (0.0125))

glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

glEnable(GL_DEPTH_TEST)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_BLEND)
glEnable(GL_NORMALIZE)

glShadeModel(GL_SMOOTH)# most obj files expect to be smooth-shaded
glEnable(GL_POINT_SMOOTH)
glEnable(GL_PROGRAM_POINT_SIZE)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

### P R O J E C T I O N ###
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
asp = width*1.0/height*1.0
glViewport(0, 0, width, height);
#// ACERTAIN IF glFrustum IS REQUIRED OR NOT AND WHA EFFECT IT HAS
#glFrustum(-1, 1, -1, 1, 4.0, 16.0) ???
gluPerspective(75.0, asp, 0.75, 512.0)
### P R O J E C T I O N ###\

syslog.syslog(syslog.LOG_ALERT,'GL params passed')
glFrontFace(GL_CW)


GEN_ERR_MSG = ''

def GEN_ERR(*args):
    global GEN_ERR_MSG
    for e in args:
        GEN_ERR_MSG += str(e)+'\n'


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
        self.mainmodel = "untitled2.obj"
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
            for vertex in self.obj.vertices:
                v0 = Vector3(vertex)
                ct = random.random() * scale_offset_max
                vertex = (v0 * (1 - ct))
                self.obj.vertices[index] = (vertex[0], vertex[1], vertex[2])
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
            vC = vB.cross(vA).normalize()
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

                for jj in range(0,2):
                    vst = list(set(vst+[int(ce) for bv in vst for ce,i in self.SUB_SUPER_MEGA[str(bv)].items()]))

                self.POLY_RADII_ARRAY[ii] = vst

                if (ii % poly_mod) == 0:
                    StackTrace("configuring radii for poly [%i of %i]" % (ii,poly_len), 'no-delay')
                    snd('system-beep', 0.3)

            snd('affirm-beep',0.5)

        fl = len(self.POLY_CENTERS_ARRAY)
        StackTrace("first_run:"+str(first_run)+" POLY_CENTERS_ARRAY", fl, self.POLY_CENTERS_ARRAY[fl - 1])
        pass

    #//build_sector_addresses: not ready to delete this ;)
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

        self.build_polygon_centers()
        StackTrace("finished building build_polygon_centers")

        self.build_stars(stars_count)
        StackTrace("finished building build_stars")


    def load_world(self, apex_nodes, level):
        self.SELECTED_APEX_VERTS = []
        Upgrade.Upgrades_All = []
        StackTrace("loading world")

        self.build_polygon_topography('nodal_random', apex_nodes, 0.02)
        #self.build_polygon_topography('arbitrary_random', 16, 0.02)
        self.obj.refresh()

        StackTrace("finished build_polygon_topography")

        self.build_polygon_centers()

        StackTrace("finished building build_polygon_centers")
        

class Score_e(dict):
    pass


class X29Element(object):
    """parent class of anything that moves and interacts with material world"""
    name = ''
    PWEs = set()
    Units_Killed = 0

    def __init__(self, elementname, radius):
        global current_level
        self.type = None
        self.PHYS = WorldPhysics()
        self.TRAC = Trace()
        self.ADDR = PolyAddressHandle(elementname)
        self.at_poly = 0
        self.mat = Matrix44()
        self.at_normal = Vector3()
        self.prev_normal = Vector3()
        self.name = elementname
        self.radius = radius
        self.init_radius = radius
        self.shield_radius = radius
        self.position = Vector3()
        self.surface_position = Vector3()
        self.altitude = 0.0
        self.collided = 0
        self.hitting = 0
        self.flaming = 0
        self.killed = 0
        self.strength = 1.0
        self.shoots = 1
        self.spawn_count = 0
        self.respawn_timer = 0
        self.dmg = 1.0
        self.htk = 12.0 #"""hits to kill"""
        self.ctk = 0.0 #"""hits"""
        self.shield = 0.0
        self.shield_init = 0.0
        self.shield_radius = radius+1.0
        self.fromlevel = current_level
        self.index = len(X29Element.PWEs)
        self.real_name = self.get_name()
        self.targeting = None
        self.rgb = (1.0,1.0,1.0)
        self.ownerclass = inspect.getmro(self.__class__)[1]
        self.ownerclass.PWEs.add(self)
        print("Created X29Element: %i %s %s" % (self.index, self.name, self.real_name))

    def get_name(self):
        aa = int(random.random() * len(names_one))-1
        bb = int(random.random() * len(names_one))-1
        return str(names_one[aa] + '-' + names_one[bb])

    def fire(self):
        #if not self.targeting: return
        active_target = None

        if self.targeting_enabled:
            active_target = self.targeting

        if self.name == 'user':
            if upgrades['laser']['on']:
                snd('laser', 0.6)
            else:
                snd('fire', 0.6)
            ftype = 'laser' if upgrades['laser']['on'] else None
            fspeed = basic_flame_speed*2.0 if upgrades['rapid_fire']['on'] else basic_flame_speed
        else:
            snd('enemy-fire',self._pos)
            ftype = None
            fspeed = basic_flame_speed

        FLAM.fire(weakref.proxy(self), self.ADDR.at_poly, active_target, fspeed, ftype)

    def flamed(self, flame_or_entt, *alt_hit):
        """when the target has been hit by a flame or impacted by another entt"""

        if alt_hit:
            """collision-driven impact dmg by half"""
            flame = None
            dmg_amt = flame_or_entt.dmg*0.25
            snd('tweak',flame_or_entt._pos)
            if self.shield > 0.0: return
        else:
            """firing-driven impact dmg by half"""
            flame = [fl for fl in FLAM.flames if fl.index == flame_or_entt][0]
            f_rel_vector = (self._pos - flame.shooter._pos)
            if f_rel_vector.length > 0: self.PHYS.Velo += (f_rel_vector.get_normalised() * flame.density) / (abs(self.mass)+0.00001)
            dmg_amt = flame.density
            if self.shield > 0.0:
                snd('tweak', self._pos)
            else:
                snd('impact', self._pos)
            self.flaming = 1
            self.hitting = 1

        fires = (10 * dmg_amt) + (20 * upgrades['insta_kill']['on'])

        if self.shield > 0.0:
            self.shield -= dmg_amt
            self.shield_radius = self.radius+(self.shield/self.shield_init)
            if flame is not None: SMOKES.gimme_a_fucking_smoke(flame._pos, int(fires/2.0), 3.0, 'fire')
        else:
            self.shield = 0
            if self.name != 'user': self.radius = self.init_radius * (1.0 - self.ctk / self.htk)
            self.shield_radius = self.radius
            SMOKES.gimme_a_fucking_smoke(self._pos, int(fires), 2.0, 'fire')
            self.ctk += dmg_amt
            self.mass = self.startmass*(1.0 - self.ctk / self.htk)


        if self.name == 'user':
            if (self.ctk / self.htk > 0.9) or (self.radius / self.init_radius < 0.11):
                self.PHYS.Velo = Vector3(0.0, 0.0, 0.0)
                self.die()
                return
            Score(scores[self.type + '-hit'] if alt_hit else scores[self.type + '-flame'], self._pos)
        else:


            if USER.PHYS.Velo.length > 8.0:
                if USER.altitude > 2.0:
                    snd('bonus', 0.5)
                    Score.msg("'Fly-By' BONUS: %d points" % scores['fly-by'])
                    Score(scores['fly-by'], self._pos)
                else:
                    snd('bonus', 0.5)
                    Score.msg("'Drive-By' BONUS: %d points" % scores['drive-by'])
                    Score(scores['drive-by'], self._pos)

            if upgrades['insta_kill']['on'] or (self.ctk / self.htk > 0.9):
                if self.killed == 0:
                    self.killed = 1
                    self.active = 0
                    delegated_flames = [f for f in FLAM.flames if f.target and f.target.name == self.name]
                    for f in delegated_flames: FLAM.flames.remove(f)

                    SMOKES.gimme_a_fucking_smoke(self._pos, 30, 1.0)
                    snd('explode', self._pos)
                    if upgrades['insta_kill']['on']:
                        SMOKES.gimme_a_fucking_smoke(self._pos, int(fires*3.0), 3.0, 'fire')
                        snd('wilhelm', self._pos)
                    Score(scores[self.type+'-kill'], self._pos, 2.0)

                    ttl = [e for e in self.ownerclass.PWEs if e.killed == 1]
                    if len(ttl) == len(self.ownerclass.PWEs)-1 and len(ttl) > 1:
                        snd('bravo')
                        Score.msg("'YOU KILLED THE WHOLE TEAM' BONUS: %d points" % scores['whole-team'])

                    self.respawn()
                    return

            Score(scores[self.type + '-hit'] if alt_hit else scores[self.type + '-flame'], self._pos)

    def hit(self, sta, *by_obj):
        """when a hit is triggered"""
        if not sta: return
        if not self.hitting:
            self.flamed(by_obj[0], 1)
            self.hitting = 1

    def respawn(self):
        global current_level
        self.ownerclass.Units_Killed += 1
        er = random.random() * 0.5 + 0.5
        self.radius = er
        self.init_radius = er
        self.shield = self.shield_init
        self.shield_radius = er + 1.0
        self.mass = er*10.0
        self.startmass = er*10.0
        self.dmg = er*current_level
        self.ctk = 0.0
        self.spawn_count += 1
        self.respawn_timer = ceil(er*1000)
        #if self.real_name in USER.entity_marker_arrow_angles: del USER.entity_marker_arrow_angles[self.real_name]

    def hold_idle(self, timepassed):
        self.PHYS.set_deccel()
        self.PHYS.update(timepassed)
        self.respawn_timer -= 1
        self.active = self.respawn_timer < 0
        if self.respawn_timer == 0:
            self.PHYS.set_position(self.ADDR.init_random_poly() * 1.2)
            self.move_entity()
        if self.active:
            GEN_ERR(self.name,'woke')
            self.killed = 0

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
        a_normal = self.ADDR.nor.copy()

        self.at_normal = a_normal


        vel = self.PHYS.Velo.length
        ldl = (current_position.length + a_height) / current_position.length

        self.surface_position = s_height * current_position
        self.zero_altitude = current_position.get_length()
        self.altitude = self.zero_altitude - self.surface_position.get_length()

        self.surface_position = ldl * current_position
        self.altitude = a_height*-1.0

        if (self.altitude < 0.075) and not self.PHYS.Jumpy:
            try:
                attitude_deg = degrees(asin(self.PHYS.Velo.get_normalised().dot(a_normal)))
            except [ValueError, ZeroDivisionError] as t_error:
                GEN_ERR(self.name,t_error)


            self.PHYS.Pos = (self.surface_position.length + 0.075) * self.surface_position.get_normalised()

            if (attitude_deg > 55.0) and (vel > 3.0):
                self.PHYS.bounce(a_normal)
                ld = (self._pos-USER._pos).length if self.name != "user" else 1.0
                snd('thud', (0.75 / ld))
                SMOKES.gimme_a_fucking_smoke(self.surface_position, 10, 1.0)
            else:
                self.PHYS.Velo *= surface_velocity_decay
                self.PHYS.creep(a_normal)

    def kill(self):
        sname = self.name
        self.__class__.PWEs.discard(self)
        print(len(self.__class__.PWEs))
        del self
        return "Deleted "+sname

    @classmethod
    def get_prox_pos(cls):
        return list(map(lambda u: get_prox(USER.mat, USER._pos, u.position), [u for u in cls.PWEs if u.active and u.name is not 'user']))

    @property
    def _pos(self):
        return self.PHYS.Pos.copy()

    @property
    def _nextpos(self):
        return self.PHYS.NextPos.copy()


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
        glColor3f(self.rgb[0],self.rgb[1],self.rgb[2])
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
        self.rgb = (1.0, 1.0, 1.0)


class PolyAddressHandle(object):
    """referenced as .ADDR"""
    def __init__(self, name):
        #self._pos = Vector3()
        #self._nextpos = Vector3()
        self.poly_a = Score_e()
        self.poly_b = Score_e()
        self.at_poly = 0
        self.last_poly = 0
        self.alt = 0
        self.nor = Vector3()
        self.message = 'None'
        self.name = name
        self.polynormavg = Vector3()
        pass

    def localize_poly(self, poly, index):
        rempoly = WORLD.POLY_CENTERS_ARRAY[index]
        poly._id = index
        poly._pos = rempoly['center'].copy()
        poly._ver = rempoly['hverts']
        poly._nea = rempoly['nearest']
        poly._nor = rempoly['normal'].copy()
        poly._set = [Vector3(WORLD.obj.vertices[poly._ver[0]]),
                     Vector3(WORLD.obj.vertices[poly._ver[1]]),
                     Vector3(WORLD.obj.vertices[poly._ver[2]])]
        return poly

    def get_avg_normal(self):
        if self.name != 'user': return
        apn = Vector3()
        for n in self.poly_a._nea:
            apn += WORLD.POLY_CENTERS_ARRAY[n]['normal']
        self.polynormavg = apn / (len(self.poly_a._nea) * -1.0)

    def init_random_poly(self):
        r = int(random.randrange(len(WORLD.POLY_CENTERS_ARRAY)))
        self.poly_a = self.localize_poly(self.poly_a, r)
        self.get_avg_normal()
        self.at_poly = r
        return self.poly_a._pos.copy()

    def init_from_poly(self, poly):
        self.poly_a = self.localize_poly(self.poly_a, poly)
        self.get_avg_normal()
        self.at_poly = poly

    def check(self, poly, pos):

        #return intersect_test(pos, world_center, poly._set[0], poly._set[1], poly._set[2])
        self.message = str(intersect_test(world_center, pos, poly._set[0], poly._set[1], poly._set[2]))

        return point_in_poly(pos, poly._set[0], poly._set[1], poly._set[2])

        #
        # try:
        #     return point_in_poly(pos, poly._set[0], poly._set[1], poly._set[2])
        #     #return point_in_poly(pos.get_normalized(), poly._set[0].get_normalised(), poly._set[1].get_normalised(), poly._set[2].get_normalised())
        # except ZeroDivisionError as err:
        #     GEN_ERR(err,pos,poly,poly._set[0], poly._set[1], poly._set[2])
        #     return (0, 0, 0)

    def get_new_poly(self, pos):
        self.last_poly = self.at_poly
        self.get_avg_normal()
        for nearc in self.poly_a._nea:
            testpoly = self.localize_poly(self.poly_b , nearc)
            d = self.check(testpoly, pos)
            if d: return [nearc, d, testpoly]
        return (0,0,0)

    def idle(self):
        p_current = self.check(self.poly_a, self._pos)

        if not p_current:
            pid, p_current, poly = self.get_new_poly(self._nextpos)
            if pid: self.poly_a = poly

        if p_current:



            self.at_poly = self.poly_a._id
            self.alt = (self._pos - self.poly_a._pos).dot(self.poly_a._nor)
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
        for npoly in WORLD.POLY_CENTERS_ARRAY[USER.ADDR.at_poly]['nearest']:
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
            special = None
            if args and len(args) > 1:
                special = None if args[1] == 'fire' else args[1]
                self.smokes.append([position + randy * 0.25,
                                    ceil(s*50.0),
                                    randy,
                                    0.2,
                                    ceil(s*100.0),
                                    position,
                                    6.0,
                                    [1.0, 1.0, 0.15, 1.0],
                                    special])
            else:
                self.smokes.append([position + randy * 0.25,
                                    1.0,
                                    randy,
                                    s,
                                    ceil(s * 60.0) * 1.0,
                                    position,
                                    mod,
                                    [0.35, 0.35, 0.35, 1.0],
                                    special])

    def show(self):
        #glFrontFace(GL_CW)

        for n in self.smokes:
            ds = sin(pi * (n[1]/n[4]))
            n[0] += (n[2] * ds) * (0.03*n[6])
            if n[0].length < n[5].length: n[0] = n[0].get_normalized()*n[5].length
            glPushMatrix()
            glFrontFace(GL_CCW)
            glTranslate(n[0].x, n[0].y, n[0].z)
            glColor4f(n[7][0],n[7][1],n[7][2],n[7][3]) #, 0.15, 0.15, 1.0)

            if n[8]:
                glMultMatrixf(USER.mat.to_opengl())
                glScale((ds),(ds),(ds))
                glCallList(n[8])
            else:
                glutSolidSphere((n[3]*ds)*self.smoke_max_size, 6, 6)


            glPopMatrix()
            n[1] += 1.0

        for n in self.smokes:
            if n[1] > n[4]-1.0: self.smokes.remove(n)


class Flame(object):
    #//FLAME WORK HERE FIRST
    flames = []

    def __init__(self):
        self.flames_max = 20
        self.firetimer = 0
        self.fireinterval = 5
        self.firespeed = 0.2
        self.flames_counter = 0
        pass

    class Bullet(object):
        """subclass of flame for granularity or granola if you prefer"""
        def __init__(self, poly, has_target):
            #global basic_weapon_duration_frames
            self.density = 0.0
            self.type = ''
            self.origin = Vector3()
            self._pos = Vector3()
            self._nextpos = Vector3()
            self.target = has_target
            self.active = 1
            self.speed = basic_flame_speed
            self.index = len(Flame.flames)
            self.rgb = (0.0,0.0,0.0)
            self.shooter = None
            self.ADDR = PolyAddressHandle('0' + str(self.index) + '-bullet')
            self.ADDR.init_from_poly(poly)
            self.frame = basic_flame_duration_frames
            self.frames = basic_flame_duration_frames
            self.expire = 0
            self.size = basic_flame_size
            self.direction = Vector3() #only used when not targeted
            Flame.flames.append(self)
            pass



        def propagate(self):
            if self.target is not None:
                d = (self.target._pos - self._pos)
                moveamt = (d.get_normalized() * self.speed) * 0.5
                if self.target.killed:
                    self.expire = 1
                    return
                if (d.length < self.target.radius) and self.active:
                    self.expire = 1
                    self.target.flamed(self.index)
                    self.target.hitting = 1
                    return
            else:
                moveamt = (self.direction * self.speed)

            self._pos += moveamt
            self._nextpos = self._pos + moveamt

            # self._pos *= 0.9999
            # self._nextpos *= 0.9999

            self.ADDR._pos = self._pos
            self.ADDR._nextpos = self._nextpos
            self.ADDR.idle()
            self.frame -= 1

            if self.frame < 0:
                self.expire = 1
                return

            try:
                hits = [ent.flamed(self.index) for ent in X29Element.PWEs
                        if not ent.killed and ent.name != self.shooter.name
                        and ent.ADDR.at_poly == self.ADDR.at_poly
                        and (ent._pos - self._pos).length < (ent.shield_radius)]
                if hits: self.expire = 1
            except ReferenceError as err:
                GEN_ERR(err, 'flame', self.index, 'orphaned')
                self.expire = 1
                return

            if self.ADDR.alt > 0:
                SMOKES.gimme_a_fucking_smoke(self._pos, 10, 2.0, 'fire')
                self.expire = 1

            if self.ADDR.last_poly != self.ADDR.at_poly:
                """check for collisions in this case"""
                #ADDR.alt!



                pass

        def draw(self):
            if self.expire:
                glPushMatrix()
                glFrontFace(GL_CW)
                glTranslate(self._pos.x, self._pos.y, self._pos.z)
                glColor4f(1.0, 1.0, 0.1, 1.0)
                glutSolidSphere(0.01, 4, 4)
                glPopMatrix()
                self.active = 0
                return
            if self.type is 'laser':
                glPushMatrix()
                glFrontFace(GL_CW)
                glColor4f(self.rgb[0], self.rgb[1], self.rgb[2], 1.0)
                glBegin(GL_LINES)
                glVertex3f(self._pos.x, self._pos.y, self._pos.z)
                glVertex3f(self.target._pos.x, self.target._pos.y, self.target._pos.z)
                glEnd()
                glTranslate(self.target._pos.x, self.target._pos.y, self.target._pos.z)
                glColor4f(1.0, 1.0, 0.1, 1.0)
                glutSolidSphere(0.1, 8, 8)
                glPopMatrix()
                self.active = 0
            elif self.type is 'basic':
                glPushMatrix()
                glFrontFace(GL_CW)
                glTranslate(self._pos.x, self._pos.y, self._pos.z)
                glColor4f(self.rgb[0], self.rgb[1], self.rgb[2], 1.0)
                e = (self.frame/float(self.frames))
                self.density = self.shooter_dmg*e
                siz = self.size*e
                glutSolidSphere(siz, 8, 8)
                glPopMatrix()







    def fire(self, shooter_obj, poly, flame_targeted, flame_speed, flame_type):
        #//CHECK FOR LINE OF SIGHT HERE? or is it enough to have the intent to fire and trace the rest on the bullet?

        flam = Flame.Bullet(poly, flame_targeted) #Score_e()
        flam.density = shooter_obj.dmg
        flam.shooter_dmg = shooter_obj.dmg
        flam.type = 'laser' if flame_type == 'laser' else 'basic'
        flam.origin = shooter_obj._nextpos
        flam._pos = shooter_obj._nextpos
        flam.active = 1
        flam.speed = flame_speed
        flam.rgb = shooter_obj.rgb
        flam.shooter = shooter_obj

        if flame_targeted is None:
            flam.direction = Vector3(shooter_obj.mat.forward)*-flame_speed
            if shooter_obj.name != 'user':
                flam.direction = (USER._pos-shooter_obj._pos).get_normalised()*flame_speed




        """get some stuff from shooter"""
        recoil = Vector3(shooter_obj.mat.forward)*0.25
        shooter_obj.PHYS.Velo += recoil



    def show(self):
        for f in self.flames:
            if f.target:
                try:
                    test_target = f.target._pos
                except ReferenceError as err:
                    f.active = 0
                    GEN_ERR(err, 'flame', f.index, 'orphaned')
                    break

                """LASER doesn't call propagate"""
                if f.type == 'laser':
                    f.draw()
                    f.target.flamed(f.index)
                    f.target.hitting = 1
                    f.active = 0

            else:
                f.propagate()
                f.draw()


        f_end_flame = [Flame.flames.remove(sf) for sf in Flame.flames if not sf.active]


class CameraHandler(object):
    """the main camera view instance"""
    def __init__(self, view_distance):
        self.transmat = Matrix44()
        self.rotmat = Matrix44()
        self.mat = Matrix44()
        self.puremat = Matrix44()
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
        self.mat = self.rotmat.copy()
        self.mat.set_row(3, (0.0,0.0,0.0,1.0))
        self.rotmat.set_row(3,self.transmat.get_row(3))
        return self.rotmat.get_inverse_rot_trans().to_opengl()
        #return self.rotmat.to_opengl()

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
        PR = (root_right.cross(UF)).normalize()
        PN = UF.cross(PR).normalize()
        PW = PN.cross(PR).normalize()
        self.rotmat.set_row(0, PN.as_tuple())
        self.rotmat.set_row(1, PR.as_tuple())
        self.rotmat.set_row(2, PW.as_tuple())

        pass

    def update_pos(self, root_position):
        """applies new camera position based upon arg."""
        self.camera_distance = camera_distance
        i = 0.05
        j = 0.05

        self.x_off += ((Vector3(USER.mat.right) - self.x_off) * i)
        #self.y_off += ((Vector3(USER.mat.up) - self.y_off) * i)
        self.y_off += ((USER.ADDR.polynormavg - self.y_off) * i)
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


class BlobEntityHandler(X29Element):
    """instance class for entities"""
    def __init__(self, name):
        er = random.random() * 0.5 + 0.5
        super(BlobEntityHandler, self).__init__(name, er)
        self.mass = er * 10.0
        self.startmass = er * 10.0
        self.radius = er
        self.init_radius = er
        self.behavior_idle = random.random()*180.0
        self.rotation_amt = 0
        self.PHYS.Velo = Vector3(0.001,0.001,0.001)
        self.active = 1
        self.graphic = entity_d(self.radius)
        self.rgb = (0.9, 0.3, 0.3)
        self.TRAC.rgb = (self.rgb[0] * 0.5, self.rgb[1] * 0.5, self.rgb[2] * 0.5)
        self.shield = 5.0
        self.shield_init = 5.0
        self.shield_radius = er + (self.shield/10.0)
        self.type = 'blob'
        self.targeting_enabled = 0

    def move_entity(self):
        self.define_mat_from_normal()
        try:
            pex = (USER._pos - self._pos)
            salt = (1.0 + (1.0*sin(self.behavior_idle)))
            e_heading = Vector3(self.mat.forward) * -salt
            e_right = Vector3(self.mat.right) * -1.0
            dir_rotate = pex.get_normalised().dot(e_right)
            rd = Vector3(0.0, dir_rotate + (salt*0.001), 0.0)
            self.apply_rotation_to_mat(rd)

            mov = e_heading * movement_speed
            mov *= movement_speed_decay

            self.PHYS.set(mov)
            self.behavior_idle += 0.5

            if self.shoots:
                try:
                    rf = int(random.random()*(20*pex.length))
                    if rf == 5 and (pex.length < 20.0):
                        #self.targeting = weakref.proxy(USER)
                        self.fire()
                except ValueError as err:
                    GEN_ERR(self.name,err,pex.length)

        except ZeroDivisionError as err:
            GEN_ERR(self.name,err,'move_entity')

        pass

    def draw_entity(self):
        pos = self._pos
        #self.graphic = entity_d(self.radius)
        #u_gl_list = glGenLists(1)
        #glNewList(u_gl_list, GL_COMPILE)
        glPushMatrix()
        glFrontFace(GL_CCW)
        glTranslate(pos.x, pos.y, pos.z)
        glMultMatrixf(self.mat.to_opengl())
        glScale(self.radius,self.radius,self.radius)

        if self.collided:
            glColor4f(0.7, 0.1, 0.7, 1.0)
            self.collided = 0
        elif self.hitting:
            glColor4f(0.8, 0.8, 0.1, 1.0)
            self.hitting = 0
        else:
            glColor4f(0.5, 0.01, 0.01, 1.0)

        glCallList(self.graphic)
        glPopMatrix()
        #glEndList()
        #return u_gl_list


class AggroEntityHandler(X29Element):
    #// WWORK HERE SECOND
    pass


class UserHandler(X29Element):
    """the main user instance"""
    def __init__(self, name):
        super(UserHandler, self).__init__(name, user_ship_scale*1.4)
        self.mass = 100.0
        self.startmass = 100.0
        self.ground_offset_height = 0.0
        self.score = 0
        self.prev_score = 0
        self.radius = 0.15
        self.init_radius = 0.15
        self.radar = 1
        self.active = 0
        self.dead = 0
        self.dead_timer = 0
        self.rgb = (0.3,1.0,0.3)
        self.dmg = 1.8
        self.ctk = 0.0
        self.htk = 25.0
        self.shield = 20.0
        self.shield_init = 20.0
        self.TRAC.rgb = (self.rgb[0]*0.5,self.rgb[1]*0.5,self.rgb[2]*0.5)
        self.type = 'user'
        self.codes = {0:'green', 1:'pink', 2:'yellow', 3:'orange', 4:'red', 5:'borked'}
        self.code = 0
        self.codered = 0
        self.targeting_enabled = 0

    def die(self):
        self.killed = 1
        off_count = 0
        for l,upgg in upgrades.items():
            off_count += int(upgg['on'])
            upgg['on'] = 0

        self.mass = 100.0
        self.startmass = 100.0

        SMOKES.gimme_a_fucking_smoke(self._pos, off_count, 0.75, upgrade)
        SMOKES.gimme_a_fucking_smoke(self._pos, 100, 1.0, 'fire')
        SMOKES.gimme_a_fucking_smoke(self._pos, 30, 1.5)
        self.dead = 1
        self.dead_timer = 100

        Score(USER.score*-0.5, USER._pos, 2.0)
        USER.score *= 0.5

        snd('wilhelm', 0.55)
        snd('nega', 0.25)
        snd('explode',0.25)

        self.code = 0
        self.codered = 0
        self.dmg = 1.0
        self.ctk = 0.0
        self.radius = self.init_radius
        self.shield = self.shield_init
        self.shield_radius = 1.0
        self.targeting_enabled = 0
        #self.entity_marker_arrow_angles = {}

    def jump(self, height):
        self.PHYS.Jumpy = 1
        self.PHYS.Velo += self.at_normal * height

    def apply_rotation(self, rotation):
        if CAM.mode == 'ground':
            #self.ground_offset_height += rotation[0]*0.25
            self.ground_offset_height = min(self.ground_offset_height+rotation[0]*0.25,1.6)
            if self.ground_offset_height < 0: self.ground_offset_height = 0.0
            rotation[0] *= 0.0
        super(UserHandler, self).apply_rotation_to_mat(rotation)


class Flag(object):
    Flags = []
    def __init__(self):
        self.id = ('flag 0%i' % (len(self.Flags)+1) )
        self.index = len(self.Flags)
        self.init_position = Vector3()
        self.position = Vector3()
        self.animation_vector = Vector3()
        self.animation_frames = float(animation_frames)
        self.animation_ct = 0.0
        self.flag_graphic = flag
        self.state = "open"
        self.poly = 0
        self.name = self.id
        self.__class__.Flags.append(self) #weakref.proxy(self))

    @classmethod
    def get_prox_pos(cls):
        return list(map(lambda u: get_prox(USER.mat, USER._pos, u.position), [u for u in cls.Flags if u.state == 'open']))

    def draw_flag(self):
        if self.state != "open":
            self.animation_ct += 1.0
            self.animation_vector = self.position.get_normalized() * 0.02 * self.animation_ct

        if self.animation_ct > self.animation_frames: self.close()

        self.position += self.animation_vector

        flag_gl_list = glGenLists(1)
        glNewList(flag_gl_list, GL_COMPILE)

        glPushMatrix()
        glTranslate(self.position.x, self.position.y, self.position.z)
        glColor4f(0.8, 0.8, 0.1, 1.0)

        glPointSize(8.0)
        ep = (self.position * 1.06) - self.position
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
        glScalef(1.5,1.5,1.5)
        glCallList(self.flag_graphic)
        glPopMatrix()
        glEndList()
        return flag_gl_list

    def close(self):
        Flag.Flags.remove(self)

    def acquire(self):
        if self.state == 'open':
            self.state = 'acquired'
            Score(scores['flag'], self.init_position)
            snd('ah')


class ScoreMarker(object):
    scores = []

    def __init__(self):
        self.message = "hello world"
        self.message_frames = 100
        self.position = Vector3
        pass

    def __call__(self, points, pos, *size):
        s = Score_e()
        s.pos = pos
        s.points = points
        s.frames = animation_frames
        s.size = 0.4
        if size: s.size *= size[0]
        self.scores.append(s)

        USER.score += points
        pass

    def msg(self,msgtxt):
        self.message = str(msgtxt)
        self.message_frames = 100

    def show(self):
        global camera_distance
        for s in self.scores:
            s.frames -= 1
            s.pos *= 1.001
            TEXTB.draw(0.5, str(s.points), s.pos, 0.0)
        r = [ScoreMarker.scores.remove(s) for s in ScoreMarker.scores if s.frames < 0]

        self.position = CAM._pos - Vector3(CAM.mat.forward)*(0.5*screen_text_depth)
        TEXTB.draw(0.6, str(int(USER.score)), self.position, 1.4)

        if self.message_frames > 0:
            TEXTB.draw(0.35, str(self.message), self.position, 1.3)
            self.message_frames -= 1


class Upgrade(object):
    Upgrades_All = []
    Cheese_Bonus_Counter = 0
    def __init__(self, upgradetype, subpoly, poly_center, poly_normal):
        self.start_height = 1.0
        self.mod_height = 0.3
        self.scale = 1.0
        self.poly = subpoly
        self.normal = Vector3(poly_normal)
        self.position = Vector3(poly_center)
        self.mod_position = Vector3()
        self.upgrade_type = upgradetype
        self.upgrade_name = upgradetype
        self.mat = Matrix44()
        self.mat.set_row(3,(self.position*self.start_height).as_tuple())
        self.define_mat_from_normal()
        self.intercepted = 0
        self.frame = 0.0
        self.index = len(self.__class__.Upgrades_All)
        self.animate_frames = animation_frames
        self.name = self.upgrade_name
        self.id = self.name+'-'+str(self.index)
        self.s_confirm = 0.0
        self.confirm = 5.0
        self.state = 0
        self.showing = 0
        self.animating = 0
        self.became_cheese = 0
        self.__class__.Upgrades_All.append(self)

    def define_mat_from_normal(self):
        PPN = self.position.get_normalized()
        UF = self.normal.normalize()
        PN = UF.cross(PPN).normalize()
        PW = PN.cross(PPN).normalize()
        self.mat.set_row(0, PN.as_tuple())
        self.mat.set_row(1, PPN.as_tuple())
        self.mat.set_row(2, PW.as_tuple())

    @classmethod
    def idle(cls):
        map(lambda u: u.show(), cls.Upgrades_All)

    @classmethod
    def get_prox_pos(cls):
        return list(map(lambda u: get_prox(USER.mat, USER._pos, u.position), cls.Upgrades_All))

    def determine_type(self):
        global current_level
        d = filter(lambda x: (upgrades[x]['lev'] <= current_level) and not upgrades[x]['on'], [k for k in upgrades for dummy in range(upgrades[k]['rarity'])])

        if len(d) > 0:
            self.upgrade_type = random.choice(d)
        else:
            s = (USER.shield / USER.shield_init)
            if s > 1.5:
                """shield-level at or > 150%"""
                self.became_cheese = 1
                self.upgrade_type = 'cheese'
                Score.msg("Upgrades and Shields at maximum. Have some cheese.")
            else:
                self.became_cheese = 0
                self.upgrade_type = 'shield'

        if USER.code >= 2: self.upgrade_type = 'systems_repair'
        self.upgrade_name = self.upgrade_type.upper().replace('_', ' ')
        self.scale = 0.5 if self.upgrade_type == 'shield' else 1.0



    def open(self):
        if self.intercepted: return
        self.state = 1
        self.s_confirm = self.confirm
        if not self.showing:
            self.determine_type()
            self.showing = 1
            if not self.animating: self.frame = self.animate_frames

    def auto_close(self):
        self.state = 0
        if not self.animating: self.frame = self.animate_frames

    def show(self):
        dscale = self.scale
        self.animating = self.frame > 0.0

        if self.animating:
            sca = (self.frame / self.animate_frames)*self.scale
            dscale = self.scale - sca if self.state == 1 else 0.0 + sca

            if self.intercepted == 1:
                self.mod_position += (USER._pos-self.mod_position)/5.0
            else:
                self.mod_position = (self.position.length + self.mod_height * dscale) * self.position.get_normalised()

            self.mat.set_row(3, self.mod_position.as_tuple())
            self.frame -= 1.0

        if self.frame == 0.0:
            self.showing = self.state
            if self.intercepted > 0:
                self.Upgrades_All.remove(self)
                return

        if self.showing:
            glPushMatrix()
            glMultMatrixf(self.mat.to_opengl())
            glScale(dscale, dscale, dscale)
            #if self.animating: glRotate((self.animate_frames-self.frame)*2.0,0.0,1.0,0.0)
            glCallList(cheese if self.became_cheese else upgrade)
            glPopMatrix()


        self.s_confirm -= 1.0
        if self.s_confirm == 0.0: self.auto_close()

    def intercept(self):
        if self.intercepted == 0:
            self.intercepted = 1
            self.auto_close()

            """if last element in upgrade queue"""
            if len(self.__class__.Upgrades_All) == 1: snd('bravo')

            upgrades[self.upgrade_type]['on'] = 1

            if self.upgrade_type == 'systems_repair':
                USER.ctk = abs(USER.ctk)
                USER.ctk -= USER.htk*0.3
                Score.msg("Systems repaired to %d%% of capacity." % int((1.0-(USER.ctk/USER.htk))*100))

            if self.upgrade_type == 'laser':
                USER.targeting_enabled = 1


            if self.upgrade_type == 'shield':
                USER.shield += USER.shield_init*0.5
                USER.shield_radius = USER.radius + (USER.shield/USER.shield_init)
                Score.msg("Shields increased to %d%%." % int((USER.shield/USER.shield_init)*100))

            if self.upgrade_type == 'strong_weapon':
                Score.msg("Weapon damage increased to %d%%." % int((USER.dmg+1.0 / USER.dmg) * 100))
                USER.dmg += 1.0

            if self.upgrade_type == 'insta_kill':
                Score.msg("Instant hot death for your opponents.")
                snd('cena',0.5)




            if self.upgrade_type == 'cheese':
                self.__class__.Cheese_Bonus_Counter += 1
                Score(2000,self.position)
                snd('cheese', 0.25)
                return



            snd('confirm', 0.25)


#//generator functions:
def generate_blob_entities(howmany):
    """the class of red blobby npcs"""
    for l in range(0, howmany):
        ENTT = BlobEntityHandler('Entity-00' + str(l))
        ENTT.index = l
        ENTT.PHYS.set_position(ENTT.ADDR.init_random_poly() * 1.5)
    StackTrace("initialized %d NPCs" % howmany)

def wipe_entities():
    while len(X29Element.PWEs):
        for el in X29Element.PWEs:
            if el.name == 'user': continue
            print(el.kill())
            break
        if len(X29Element.PWEs) == 1:break

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
    if entity.shield == 0:
        glColor4f(0.2, 0.8, 0.2, 1.0)
    else:
        glColor4f(0.1, 0.6, 0.1, 1.0)
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
    return (angle, pdx.length)

def physical_world_element_collisions():
    """handle general collisions on frame
    to avoid redundancies"""
    def collide(clsA, clsB):
        x = (clsA._nextpos - clsB._nextpos)
        if not x.length: return
        mtd = x * (((clsA.shield_radius + clsB.shield_radius) - x.length) / x.length)
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
    for e in X29Element.PWEs:
        #trace of collisions with this specific entry
        sel = weakref.proxy(e)
        element_collision = [weakref.proxy(el) for el in X29Element.PWEs if el.active and (el.name != sel.name) and ((el._nextpos-sel._nextpos).length < (sel.shield_radius + el.shield_radius))]
        for el in element_collision:
            if (el,sel) not in res:
                el.collided = 1 if sel.name == 'user' else 0
                sel.collided = 1 if el.name == 'user' else 0
                el.hit(el.collided, sel)
                sel.hit(sel.collided, el)
                collide(sel,el)
                res.append((sel,el))

def target_entity(user_obj):
    #if ent.name != user_obj.name
    targeted = [((ent._pos-user_obj._pos).length, weakref.proxy(ent)) for ent in X29Element.PWEs if ent.active and BOX.testBounds(ent._pos)]
    return min(targeted)[1] if len(targeted) else 0

#//drawing functions
def draw_user():
    pos = USER._pos
    u_gl_list = glGenLists(1)
    glNewList(u_gl_list, GL_COMPILE)
    glPushMatrix()

    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glScale(user_ship_scale, user_ship_scale, user_ship_scale)
    glCallList(pyramid)
    glPopMatrix()

    #
    glPushMatrix()
    glTranslate(pos.x, pos.y, pos.z)
    glMultMatrixf(USER.mat.to_opengl())
    glTranslate(0.0, 0.0, -0.5)
    glScale(user_arrow_scale, user_arrow_scale, user_arrow_scale)
    glCallList(arrow)
    glPopMatrix()

    if USER.radar:

        if upgrades['upgrade_radar']['on']:
            for angle in Upgrade.get_prox_pos():
                glColor4f(0.85, 0.65, 0.0, 1.0)
                siz = 5.0 * (10.0 / angle[1])
                pointsize = siz if siz < 4.0 else 4.0
                glPointSize(pointsize)
                glPushMatrix()
                glTranslate(pos.x, pos.y, pos.z)
                glMultMatrixf(USER.mat.to_opengl())
                glRotate(degrees(angle[0]), 0.0, 1.0, 0.0)
                glTranslate(0.0, 0.0, -1.0) # + (0.5 / angle[1]))
                glBegin(GL_POINTS)
                glVertex3f(0.0, 0.0, 0.0)
                glEnd()
                glPopMatrix()

        if upgrades['flag_radar']['on']:
            for angle in Flag.get_prox_pos():
                siz = 10.0 * (10.0 / angle[1])
                pointsize = siz if siz < 10.0 else 10.0
                glPushMatrix()
                glTranslate(pos.x, pos.y, pos.z)
                glMultMatrixf(USER.mat.to_opengl())
                glPointSize(pointsize)
                glColor4f(1.0, 1.0, 0.0, 1.0)
                glRotate(degrees(angle[0]), 0.0, 1.0, 0.0)
                glTranslate(0.0, 0.0, -2.0 + (2.0 / angle[1]))
                glBegin(GL_POINTS)
                glVertex3f(0.0, 0.0, 0.0)
                glEnd()
                glScale(0.2, 0.2, 0.2)
                glCallList(flag)
                glPopMatrix()

        if upgrades['enemy_radar']['on']:
            for angle in X29Element.get_prox_pos():
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


    if USER.shield > 0.0:
        glPushMatrix()
        #glFrontFace(GL_CCW)

        glTranslate(pos.x, pos.y, pos.z)
        glColor4f(0.1, 0.6, 0.1, 0.1)

        if USER.hitting:
            glColor4f(0.2, 0.7, 0.2, 0.3)
            USER.hitting = 0

        glutSolidSphere(USER.shield_radius,24,24)
        glPopMatrix()

    glPopMatrix()
    glEndList(u_gl_list)
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

def draw_error_gon():
    pass

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

### M O D E L V I E W ###
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
### M O D E L V I E W ###
TEXTB = BetterTypeHandler()

def arrow_d():
    arrow = glGenLists(1)
    glNewList(arrow, GL_COMPILE)
    glColor4f(0.0, 0.55, 0.0, 0.5)
    glPushMatrix()
    #glFrontFace(GL_CCW)

    glBegin(GL_POLYGON)
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, -3.0)
    glVertex3f(2.5, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 2.0)
    glVertex3f(-1.0, 0.0, 2.0)
    glVertex3f(-1.0, 0.0, 0.0)
    glVertex3f(-2.5, 0.0, 0.0)
    glVertex3f(0.0, 0.0, -3.0)

    # glVertex3f(0.0, 0.0, -3.0)
    # glVertex3f(-2.5, 0.0, 0.0)
    # glVertex3f(-1.0, 0.0, 0.0)
    # glVertex3f(-1.0, 0.0, 2.0)
    # glVertex3f(1.0, 0.0, 2.0)
    # glVertex3f(1.0, 0.0, 0.0)
    # glVertex3f(2.5, 0.0, 0.0)
    # glVertex3f(0.0, 0.0, -3.0)

    glEnd()
    glPopMatrix()
    glEndList()
    return arrow
arrow = arrow_d() #the user arrow-marker icon
def pyramid_d():
    pyramid = glGenLists(1)
    glNewList(pyramid, GL_COMPILE)
    glPushMatrix()
    glFrontFace(GL_CW)

    glColor4f(0.1, 0.55, 0.1, 1.0)
    h = float(2.28824561127089)
    s = float(1.41421356237309)

    glBegin(GL_POLYGON)
    glVertex3f(0.0, 0.0, s)
    glVertex3f(s, 0.0, 0.0)
    glVertex3f(0.0, 0.0, -s)
    glVertex3f(-s, 0.0, 0.0)
    glEnd()


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

    glPopMatrix()
    glEndList()
    return pyramid
pyramid = pyramid_d() #the user icon
def bar_d():
    bar = glGenLists(1)
    glNewList(bar, GL_COMPILE)
    #glFrontFace(GL_CW)
    glColor4f(0.6, 0.4, 0.0, 1.0)
    Nm = 1.0
    glBegin(GL_POLYGON)
    a = Vector3(0.1, 0.3, -0.1)
    b = Vector3(0.1, -0.3, -0.1)
    c = Vector3(-0.1, -0.3, -0.1)
    d = Vector3(-0.1, 0.3, -0.1)
    N = getNormal(a, b, c)*Nm
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
    N = getNormal(a, b, c)*Nm
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
    N = getNormal(a, b, c)*Nm
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
    N = getNormal(a, b, c)*Nm
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
    N = getNormal(a, b, c)*Nm
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
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glEndList()
    return bar
bar = bar_d() #upgrade ICON component - may be drawn incorrectly
def upgrade_d():
    upgrade = glGenLists(1)
    glNewList(upgrade, GL_COMPILE)
    glPushMatrix()
    glFrontFace(GL_CCW)
    glColor4f(0.5, 0.5, 0.5, 1.0)
    glCallList(bar)
    glRotate(90.0, 0.0, 0.0, 1.0)
    glCallList(bar)
    glRotate(90.0, 1.0, 0.0, 0.0)
    glCallList(bar)
    glPopMatrix()
    glEndList()
    return upgrade
upgrade = upgrade_d() #upgrade ICON
def entity_d(radius):
    entity = glGenLists(1)
    glNewList(entity, GL_COMPILE)
    #glFrontFace(GL_CW)
    glutSolidSphere(radius, 12, 12)
    glTranslate(radius * 0.4, 0.0, radius * -0.9)
    glutSolidSphere(radius * 0.2, 8, 8)
    glTranslate(radius *-0.8, 0.0, 0.0)
    glutSolidSphere(radius * 0.2, 8, 8)
    glTranslate(radius * 0.4, 0.0, radius * 1.1)
    glEndList()
    return entity
def target_d(radius):
    target_list = glGenLists(1)
    glNewList(target_list, GL_COMPILE)
    glutWireCube(radius)
    glEndList()
    return target_list
def flag_d():
    flag_list = glGenLists(1)
    glNewList(flag_list, GL_COMPILE)
    glPushMatrix()
    glFrontFace(GL_CW)
    glColor4f(0.6, 0.6, 0.0, 1.0)

    h = 1.0
    w = h * 0.0375
    lw = h * 0.075
    n = -1.0

    #flag top north
    glBegin(GL_POLYGON)
    a = Vector3(0.0, h, w)
    b = Vector3(h * 0.5, h, w)
    c = Vector3(h * 0.4, h * 0.8, w)
    d = Vector3(h * 0.5, h * 0.6, w)
    e = Vector3(0.0, h * 0.6, w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glVertex3f(e.x, e.y, e.z)
    glEnd()

    #flag leg north
    glBegin(GL_POLYGON)
    a = Vector3(0.0, h, w)
    b = Vector3(lw, h, w)
    c = Vector3(lw, 0.0, w)
    d = Vector3(0.0, 0.0, w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag side east
    glBegin(GL_POLYGON)
    a = Vector3(0.0, h, w)
    b = Vector3(0.0, 0.0, w)
    c = Vector3(0.0, 0.0, -w)
    d = Vector3(0.0, h, -w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag top south
    glBegin(GL_POLYGON)
    a = Vector3(0.0, h * 0.6, -w)
    b = Vector3(h * 0.5, h * 0.6, -w)
    c = Vector3(h * 0.4, h * 0.8, -w)
    d = Vector3(h * 0.5, h, -w)
    e = Vector3(0.0, h, -w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glVertex3f(e.x, e.y, e.z)
    glEnd()

    #flag leg south
    glBegin(GL_POLYGON)
    a = Vector3(0.0, h, -w)
    b = Vector3(0.0, 0.0, -w)
    c = Vector3(lw, 0.0, -w)
    d = Vector3(lw, h, -w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag side west
    glBegin(GL_POLYGON)
    a = Vector3(lw, h * 0.6, -w)
    b = Vector3(lw, 0.0, -w)
    c = Vector3(lw, 0.0, w)
    d = Vector3(lw, h * 0.6, w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag top bottom
    glBegin(GL_POLYGON)
    a = Vector3(h * 0.5, h * 0.6, -w)
    b = Vector3(lw, h * 0.6, -w)
    c = Vector3(lw, h * 0.6, w)
    d = Vector3(h * 0.5, h * 0.6, w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag edge bottom
    glBegin(GL_POLYGON)
    a = Vector3(h * 0.5, h * 0.6, w)
    b = Vector3(h * 0.4, h * 0.8, w)
    c = Vector3(h * 0.4, h * 0.8, -w)
    d = Vector3(h * 0.5, h * 0.6, -w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    #flag edge top
    glBegin(GL_POLYGON)
    a = Vector3(h * 0.5, h, -w)
    b = Vector3(h * 0.4, h * 0.8, -w)
    c = Vector3(h * 0.4, h * 0.8, w)
    d = Vector3(h * 0.5, h, w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    # #flag top top
    glBegin(GL_POLYGON)
    a = Vector3(h * 0.5, h, -w)
    b = Vector3(h * 0.5, h, w)
    c = Vector3(0.0, h, w)
    d = Vector3(0.0, h, -w)
    N = getNormal(a, b, c)*n
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    glPopMatrix()
    glEndList()
    return flag_list
flag = flag_d() #flag ICON
def cheese_d():
    cheese = glGenLists(1)
    glNewList(cheese, GL_COMPILE)
    glPushMatrix()
    c_w = 0.5
    c_l = 0.75
    c_h = 0.5
    Nm = -1.0
    glColor4f(0.8, 0.5, 0.1, 1.0)

    # cheese top
    glBegin(GL_POLYGON)
    a = Vector3(0.0, c_h * 0.5, c_l * -0.5)
    b = Vector3(c_w * 0.5, c_h * 0.5, c_l * 0.5)
    c = Vector3(c_w * -0.5, c_h * 0.5, c_l * 0.5)
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()

    # cheese back edage
    glBegin(GL_POLYGON)
    a = Vector3(c_w * 0.5, c_h * 0.5, c_l * 0.5)
    b = Vector3(c_w * 0.5, c_h * -0.5, c_l * 0.5)
    c = Vector3(c_w * -0.5, c_h * -0.5, c_l * 0.5)
    d = Vector3(c_w * -0.5, c_h * 0.5, c_l * 0.5)
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    # cheese right edage
    glBegin(GL_POLYGON)
    a = Vector3(c_w * -0.5, c_h * 0.5, c_l * 0.5)
    b = Vector3(c_w * -0.5, c_h * -0.5, c_l * 0.5)
    c = Vector3(0.0, c_h * -0.5, c_l * -0.5)
    d = Vector3(0.0, c_h * 0.5, c_l * -0.5)
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    # cheese left edage
    glBegin(GL_POLYGON)
    a = Vector3(0.0, c_h * 0.5, c_l * -0.5)
    b = Vector3(0.0, c_h * -0.5, c_l * -0.5)
    c = Vector3(c_w * 0.5, c_h * -0.5, c_l * 0.5)
    d = Vector3(c_w * 0.5, c_h * 0.5, c_l * 0.5)
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glVertex3f(d.x, d.y, d.z)
    glEnd()

    # cheese bottom
    glBegin(GL_POLYGON)
    a = Vector3(0.0, c_h * -0.5, c_l * -0.5)
    b = Vector3(c_w * -0.5, c_h * -0.5, c_l * 0.5)
    c = Vector3(c_w * 0.5, c_h * -0.5, c_l * 0.5)
    N = getNormal(a, b, c)*Nm
    glNormal3f(N.x, N.y, N.z)
    glVertex3f(a.x, a.y, a.z)
    glVertex3f(b.x, b.y, b.z)
    glVertex3f(c.x, c.y, c.z)
    glEnd()



    glPopMatrix()
    glEndList()
    return cheese
cheese = cheese_d() #cheese ICON

current_level = 0
star_gl_list = []
user_message = 'WELCOME'
world_center = Vector3(0.0,0.0,0.0)
time_passed = clock.tick()
WORLD = WorldModelSet()
SMOKES = Corey_Has_Smokes()
FLAM = Flame()
CAM = CameraHandler(camera_distance)
USER = UserHandler('user')
ship_direction_vector = Vector3()
rotation_direction = Vector3()
movement_direction = Vector3()
COR = 1.0
attitude_deg = 0
N = 1.0 #camera_distance*0.25
F = 12.0 #camera_distance*8
BOX = ViewClipBox(N, F, float(width)/(F), float(height)/(F), 0.1)
Score = ScoreMarker()
ship_direction_vector.set(0.0, 0.0, 0.0)
key_rep_frame = 0
control_vectors = {}
stat_trigger = 0

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

def fire(*bypass):
    global flame_timer
    if upgrades['rapid_fire']['on'] or len(bypass):
        if flame_timer == 0:
            if USER.targeting_enabled:
                if USER.targeting:
                    USER.fire()
                else:
                    snd('beep', 0.2)
            else:
                USER.fire()
            flame_timer = flame_rep_max
        else:
            flame_timer -= 1

    return fire.__name__
    pass

def fire_aux():
    """for single key fire only"""
    global flame_timer
    flame_timer = 0
    fire('bypass')
    return fire_aux.__name__
    pass

def help_info():
    run_x29.info_trigger = 1
    return help_info.__name__

def zoom_in():
    global camera_distance
    if camera_distance > camera_distance_min: camera_distance *= 0.9
    return zoom_in.__name__

def zoom_out():
    global camera_distance
    if camera_distance < camera_distance_max: camera_distance *= 1.1
    return zoom_out.__name__

#//separated runtime functions
def x29_first_run():
    """Setup the game"""
    global star_gl_list
    CAM_POS = Vector3(0.0, 0.0, 0.0)
    CAM.update_pos(CAM_POS)
    glLoadMatrixf(CAM.get_view())
    WORLD.init_world_model_bases()
    star_gl_list = draw_stars()
    snd('welcome')

def x29_world_init():
    global movement_speed, init_world_size, current_level
    """Setup the game level, clean first"""
    Upgrade.Cheese_Bonus_Counter = 0
    X29Element.Units_Killed = 0
    WORLD.load_world(init_world_size, current_level)
    StackTrace('WORLD.load_world(%s) COMPLETE' % init_world_size)
    USER.PHYS.set_position(USER.ADDR.init_random_poly() * 1.5)

    #UPGRADES POLICE
    Upgrade.Upgrades_All = []
    for st in range(0, 20):
        r_poly = int(random.randrange(0, len(WORLD.POLY_CENTERS_ARRAY)))
        r_vert = WORLD.POLY_CENTERS_ARRAY[r_poly]['hverts'][0]
        Upos = WORLD.POLY_CENTERS_ARRAY[r_poly]['center']
        Unorm = WORLD.POLY_CENTERS_ARRAY[r_poly]['normal']
        Upgrade('blank', r_vert, Upos, Unorm)
    StackTrace("finished adding %d upgrades" % 20)

    #FLAGS POLICE
    max_heights = []
    Flag.Flags = []
    flag_ctr = 0
    for vert in WORLD.SELECTED_APEX_VERTS:
        if flag_ctr <= current_level + 1:
            FLAG = Flag()
            FLAG.poly = vert
            FLAG.position = Vector3(WORLD.obj.vertices[vert])
            FLAG.init_position = Vector3(WORLD.obj.vertices[vert])
            max_heights.append(FLAG.position.length)
            flag_ctr += 1
        else: break
    WORLD.model_world_max_radius = max(max_heights)
    StackTrace("finished adding %d flags" % flag_ctr)

    #ENTITY POLICE
    wipe_entities()
    entct = min([int((current_level+1)+(random.random()*4)),10])
    generate_blob_entities(entct)
    StackTrace('finished spawning %d blobs' % entct)

    StackTrace.get_time()

    movement_speed += 0.3
    init_world_size += 3
    current_level += 1

    return 'ok'

tsec = 0.0
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def x29_runtime(timedelta, has_input):
    state = 1
    global rotation_direction, movement_direction, ship_direction_vector, star_gl_list, GEN_ERR_MSG, tsec

    glCallList(WORLD.obj.gl_list)
    glCallList(star_gl_list)
    # glCallList(BOX.showBounds())
    light_d = USER._pos + Vector3(USER.mat.up) * 2.0
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 100, 0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (light_d.x, light_d.y, light_d.z, 1.0))
    # glCallList(draw_world_user_line(USER._pos, USER.surface_position))
    # glCallList(draw_world_user_line(USER._pos, USER.surface_position))
    # glCallList(draw_utility_line(USER._pos, USER._pos + USER.at_normal * 2.0, (1.0, 0.0, 1.0)))
    # glCallList(draw_utility_line(USER._pos, USER._pos + USER.ADDR.polynormavg * 2.0, (1.0, 1.0, 0.0)))
    # glCallList(draw_utility_line(USER._pos, USER._pos+Vector3(USER.mat.up) * 1.5, (1.0,1.0,1.0)))
    # glCallList(draw_utility_line(USER._pos, USER._pos+Vector3(CAM.rotmat.up) * 1.0, (1.0,0.0,0.0)))
    glCallList(draw_sector_poly_hilight(USER.ADDR.at_poly))
    glCallList(draw_sub_sector_verts(USER.ADDR.at_poly))
    glCallList(draw_poly_hilight_group(USER.ADDR.at_poly))


    Score.show()

    if run_x29.info:
        inst_position = (CAM._pos - Vector3(CAM.mat.forward)*4.0)+Vector3(CAM.mat.right)*2.0
        since_dbg = 1
        if since_dbg:
            obs = {'UALT':str(USER.altitude)[0:6],
                   'UALZ':str(USER.ADDR.message),
                   'GFPS':str(clock.get_fps())[0:2],
                   'UPOL':USER.ADDR.at_poly,
                   'CMOD':CAM.mode,
                   'UTGT':USER.targeting.real_name if USER.targeting else 'NO target',
                   'URAD':USER.shield_radius,
                   'UCTK':USER.ctk,
                   'GENE':GEN_ERR_MSG,
                   'TSEC':timedelta}
            st = [kk+':  '+str(kv) for kk,kv in obs.items()]
            st = '\n'.join(st)
        else:
            st = instructions

        TEXTB.draw(0.25, st, inst_position, 1.0, 'left')
        if run_x29.info == 2:
            run_x29.info = 0


    tsec += timedelta
    if tsec > 10:
        tsec = 0.0
        GEN_ERR_MSG = ''


    la = len(WORLD.POLY_CENTERS_ARRAY)-1
    e = WORLD.POLY_CENTERS_ARRAY[la]
    glCallList(draw_sector_poly_hilight(la))
    TEXTB.draw(0.6, "LAST SECTOR", e['center'], 0.5)

    fi = 0
    e = WORLD.POLY_CENTERS_ARRAY[fi]
    glCallList(draw_sector_poly_hilight(fi))
    TEXTB.draw(0.6, "FIRST SECTOR", e['center'], 0.5)

    #for f in Flame.flames:glCallList(draw_sector_poly_hilight(f.ADDR.at_poly))


    if TRACE_ENABLE: glCallList(USER.TRAC.draw(USER._pos))
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

    USER.PHYS.set(ship_direction_vector*(1.0+(upgrades['traction_control']['on']*1.0)))
    USER.PHYS.Accel *= (1.0-(upgrades['traction_control']['on']*0.95))

    USER.update(timedelta)
    CAM.update_pos(USER._pos)
    glLoadMatrixf(CAM.get_view())

    if len(X29Element.PWEs) > 1: physical_world_element_collisions()

    """NPCs LIST"""
    for BLOB in X29Element.PWEs:
        if BLOB.name == 'user':
            pass
        elif BLOB.active:
            BLOB.update(timedelta)
            BLOB.move_entity()
            if upgrades['enemy_radar']['on'] and USER.radar:
                dx = (BLOB._pos - USER._pos)
                if dx.length < 10.0:
                    dist = str(int(floor(dx.length))) + 'm'
                    de = dx.cross(Vector3(USER.mat.up))
                    df = de.cross(Vector3(USER.mat.up))
                    TEXTB.draw(0.6, dist, USER._pos - (df.normalize()*2.0), 0.0)
            BLOB.draw_entity()
            if TRACE_ENABLE: glCallList(BLOB.TRAC.draw(BLOB._pos))
        else:
            BLOB.hold_idle(timedelta)

    """FLAGS LIST"""
    for fl in Flag.Flags:
        glCallList(fl.draw_flag())
        fx = (fl.position - USER._pos)
        if fl.poly in WORLD.POLY_RADII_ARRAY[USER.ADDR.at_poly]:
            thr = USER._pos.length > fl.position.length
            TEXTB.draw(0.8, fl.id, fl.position, 1.5)
            if fl.state == "open" and upgrades['flag_magnet']['on'] and (fx.length < 16.0) and thr:
                glCallList(draw_utility_line(USER._pos,fl.position,(1.0,1.0,0.0)))
            if upgrades['flag_magnet']['on'] and (fx.length < 6.0) and thr:
                fl.acquire()
            elif fx.length < 0.5:
                Score.msg("Acquired '%s' of %d!" % (fl.id, len(Flag.Flags)))
                fl.acquire()

    """UPGRADES LIST"""
    ups = [u for u in Upgrade.Upgrades_All if u.poly in WORLD.POLY_RADII_ARRAY[USER.ADDR.at_poly]]
    for upg in ups:
        """trigger open for upg"""
        upg.open()
        ux = (upg.position - USER._pos)
        if ux.length < 8.0:
            TEXTB.draw(0.5, upg.upgrade_name, upg.position, 0.5*upg.scale)
            if ux.length < 0.5+USER.radius:
                upg.intercept()

    Upgrade.idle()

    """TARGETING"""
    if USER.targeting_enabled:
        N_look = USER._pos + Vector3(USER.mat.forward) * -1.0  #camera_distance
        BOX.get_ground_interfere()
        BOX.setClipBounds(USER._pos, N_look, Vector3(USER.mat.up))
        target = target_entity(USER)

        if target:
            glCallList(target_acquired(target))
            USER.targeting = target
            TEXTB.draw(1.0, target.real_name + ' ' + str(target.spawn_count), target._nextpos, target.radius+0.2)
            if target.flaming:
                mms = str(int((target.radius / target.init_radius) * 100)) + '%'
                TEXTB.draw(2.0, mms, target._nextpos, target.radius+0.4)
        else:
            USER.targeting = None

    FLAM.show()
    SMOKES.show()

    if USER.hitting:
        sms = int((1.0 - (USER.ctk / USER.htk)) * 100)
        if USER.shield > 0.0:
            mms = (int((USER.shield / USER.shield_init) * 100))
            if USER.code == 0:
                Score.msg("Shields at %d%%" % mms)
            else:
                Score.msg("Shields at %d%%. Systems at %d%% [CODE %s]" % (mms, sms, USER.codes[USER.code].upper()))
        else:
            prev_code = USER.code
            USER.code = int(5.0-ceil(sms/25.0))
            if prev_code != USER.code:
                if (USER.code > prev_code): snd('code-alert')
                if sms < 100.0:
                    if (USER.code < prev_code): snd('affirm-beep')
            if USER.code < 0: USER.code = 0
            if USER.code == 4 and USER.codered == 0:
                snd('code-red', 0.75, -1)
                USER.codered = 1
            Score.msg("ISSUE: shields down, systems integrity at %d%% [CODE %s]" % (sms, USER.codes[USER.code].upper()))
            if USER.code >= 4: SMOKES.gimme_a_fucking_smoke(USER._pos, 4, 1.0)
        if USER.code < 4:
            USER.codered = 0
            snd('code-red', 'stop')
    if USER.dead:
        Score.msg("U DEAD. Score reduced by 50%% to %i. Upgrades reset. Sorry." % int(USER.score))
        USER.dead_timer -= 1
        if USER.dead_timer == 0:
            Score.msg("Welcome back to Space Program x29")
            USER.dead = 0
            USER.killed = 0
            USER.PHYS.set_position(USER.ADDR.init_random_poly() * 1.5)
    else:
        pass
        glCallList(draw_user())


    #//NPCs FOLLOW-UP LIST FOR NPCS SHIELDS
    for BLOB in X29Element.PWEs:
        if BLOB.active and (BLOB.shield > 0.0):
            #and not BLOB.killed
            glPushMatrix()
            glFrontFace(GL_CCW)
            glColor4f(0.1, 0.6, 0.1, 0.1)
            if BLOB.hitting:
                glColor4f(0.2, 0.7, 0.2, 0.3)
                BLOB.hitting = 0
            sh = BLOB.shield_radius
            glTranslate(BLOB._pos.x,BLOB._pos.y,BLOB._pos.z)
            glutSolidSphere(sh, 24, 24)
            glPopMatrix()


    """level flip"""
    if len(Flag.Flags) == 0: state = 0
    return state
    pass
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def x29_level_transition():
    #snd('james-gang',0.25)

    for i in range(0, 100):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        TEXTB.draw(1.0, str("LEVEL %d CLEARED!" % current_level), run_x29.RENDER_POS, 0.45)
        sca = USER.prev_score + int(i*(USER.score/100))
        scb = int(USER.score - USER.prev_score)
        score_str = "previous score: %i | points earned this level: %i" % (USER.prev_score, scb)
        TEXTB.draw(0.5, score_str, run_x29.RENDER_POS, -0.25)
        TEXTB.draw(1.4, sca, run_x29.RENDER_POS, 0.0)
        snd('ticker')
        pygame.display.flip()

    snd('ticker', 'stop')
    snd('morphin', 0.45)
    pygame.time.wait(2000)


    bonuses = {'level':{'tot': 10000, 'mod':current_level, 'msg':'level multiplier'},
                'cheese':{'tot': 2000, 'mod':Upgrade.Cheese_Bonus_Counter, 'msg':'collected cheeses multiplier'},
                'upgrade':{'tot': 50000, 'mod':int(len(Upgrade.Upgrades_All) == 0), 'msg':'all world upgrades collected. wow.'},
                'lethality': {'tot': 1000, 'mod':X29Element.Units_Killed, 'msg': 'units killed'}}

    for bon,dat in bonuses.items():
        if dat['mod'] > 0:
            title_str = "%s BONUS" %(bon.upper())
            message_str = "%s [%i] x %i" %(dat['msg'].upper(), dat['mod'], dat['tot'])
            mod_score = dat['mod']*dat['tot']
            for i in range(0, 50):
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                TEXTB.draw(1.0, title_str, run_x29.RENDER_POS, 0.45)
                sca = int(i * (mod_score / 100))
                USER.score += sca
                TEXTB.draw(0.5, message_str, run_x29.RENDER_POS, -0.25)
                TEXTB.draw(1.4, USER.score, run_x29.RENDER_POS, 0.0)
                snd('ticker')
                pygame.display.flip()
            snd('ticker', 'stop')
            snd('morphin', 0.45)
            pygame.time.wait(1500)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    TEXTB.draw(1.4, "PREPARE YOUR SPACE", run_x29.RENDER_POS, 0.0)
    pygame.display.flip()
    pygame.time.wait(1000)
    USER.prev_score = USER.score
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

def quit_x29():
    StackTrace("quitting")
    pygame.quit()
    exit()

def run_x29(running):
    StackTrace('run_x29 STARTED')
    run_x29.info_trigger = 0
    run_x29.info = 0
    run_x29.stat_trigger = 0
    run_x29.pause_trigger = 0
    run_x29.paused = 0
    run_x29.RENDER_POS = Vector3
    clock.tick()
    global rotation_direction, movement_direction, movement_speed, init_world_size, current_level, GEN_ERR_MSG

    while running:
        run_x29.RENDER_POS = CAM._pos - Vector3(CAM.mat.forward) * screen_text_depth
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
            USER.die()
            print('OFF', USER.ground_offset_height)
            print('FPS', str(clock.get_fps())[0:4])
            print('CAM', str(camera_distance))
            run_x29.stat_trigger = 0

        if run_x29.info_trigger:
            run_x29.info += 1
            run_x29.info_trigger = 0

        if run_x29.pause_trigger:
            run_x29.paused += 1
            run_x29.pause_trigger = 0

        if run_x29.paused:
            TEXTB.draw(2.0, str('Space Program x29 PAUSED'), run_x29.RENDER_POS, 0.0)
            if run_x29.paused == 2:
                run_x29.paused = 0
        else:
            state = x29_runtime(time_passed_seconds,has_input)
            if not state: return "stop"



        pygame.display.flip()

def check_all_refs(scope_name, do_del):
    current_module = sys.modules[scope_name]
    print(current_module)
    gc.collect()
    for name, obj in inspect.getmembers(sys.modules[scope_name]):
        if inspect.isclass(obj):
            if obj.__module__ == scope_name:
                print(name, obj)
                for obj_record in gc.get_objects():
                    if isinstance(obj_record, obj):
                        print(name, obj_record)
                        try:
                            print(obj_record.name, obj_record.index)
                        except AttributeError:
                            pass
                print('+++++++++++')

if __name__ == "__main__":
    x29_first_run()
    x29_world_init()
    RUNCOUNTER = 0
    CAM.set_mode('ground')
    while True:
        StackTrace("S T A R T  P R O G R A M  x 2 9 - %d" % RUNCOUNTER)
        c = run_x29(True)
        if c == "stop":
            #check_all_refs(__name__, 0)
            st = x29_level_transition()
            st = x29_world_init()
            clock.tick()
            RUNCOUNTER += 1
            continue
        StackTrace("D O N E")
        break





#//RBOUD 3500 LINES
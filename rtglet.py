from pyglet.gl import *
import numpy as NP
from gameobjects.matrix44 import *
from gameobjects.vector3 import *
from OpenGL.GLUT import *
import ctypes

window = pyglet.window.Window()


# light_pos = ctypes.POINTER(ctypes.c_float * 4)((0.0, 300.0, 0.0, 1.0))
# print(light_pos)

def vec(*args):
    return (GLfloat * len(args))(*args)

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, vec(0.0, 300.0, 0.0, 1.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.5, 0.5, 0.5, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.48, 0.48, 0.48, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

glEnable(GL_DEPTH_TEST)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_BLEND)
glEnable(GL_NORMALIZE)

glShadeModel(GL_SMOOTH)# most obj files expect to be smooth-shaded
glEnable(GL_POINT_SMOOTH)
glEnable(GL_PROGRAM_POINT_SIZE)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# ### P R O J E C T I O N ###
# glMatrixMode(GL_PROJECTION)
# glLoadIdentity()
# asp = window.width*1.0/window.height*1.0
# glViewport(0, 0, window.width, window.height)
# #// ACERTAIN IF glFrustum IS REQUIRED OR NOT AND WHA EFFECT IT HAS
# #glFrustum(-1, 1, -1, 1, 4.0, 16.0) ???
# gluPerspective(75.0, asp, 0.75, 512.0)
# ### P R O J E C T I O N ###\
# glFrontFace(GL_CW)


# vertices = [
#     0, 0,
#     window.width, 0,
#     window.width, window.height]
# vertices_gl = (GLfloat * len(vertices))(*vertices)
#
# glEnableClientState(GL_VERTEX_ARRAY)
# glVertexPointer(2, GL_FLOAT, 0, vertices_gl)



glMatrixMode(GL_MODELVIEW)

#//separated runtime functions
def getNormal(v1, v2, v3):
   a = v1 - v2
   b = v1 - v3
   return a.cross(b).normalize()

def pyramid_d():
    pyramid = glGenLists(1)
    glNewList(pyramid, GL_COMPILE)
    glPushMatrix()


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





@window.event
def on_key_press(symbol, modifiers):
    print(str(symbol),symbol)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glColor4f(0.1, 0.55, 0.1, 1.0)
    glutSolidSphere(1.0, 8, 8)
    # glTranslatef(0.0,0.0,-10.0)
    # #glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 2)
    # glCallList(pyramid)
    # glTranslatef(0.0,0.0,20.0)
    # glCallList(pyramid)



if __name__ == "__main__":
    pyglet.app.run()
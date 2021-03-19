import numpy
from freetype import *
import OpenGL.GL as gl
import OpenGL.GLUT as glut

base, texid = 0, 0
text  = '''gl.glTexCoord2f( ((x+n)*dx), \n (y  )*dy ), gl.glVertex( w, 0 )'''
widths = []

def on_display( ):
    global texid
    gl.glClearColor(1,1,1,1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glBindTexture( gl.GL_TEXTURE_2D, texid )
    gl.glColor(0,0.1,0,0.85)
    gl.glPushMatrix( )
    gl.glTranslate( 10, 100, 0 )
    gl.glPushMatrix( )
    gl.glListBase( base+1 )
    gl.glCallLists( [ord(c) for c in text] )
    gl.glPopMatrix( )
    gl.glPopMatrix( )
    glut.glutSwapBuffers( )

def on_reshape( width, height ):
    gl.glViewport( 0, 0, width, height )
    gl.glMatrixMode( gl.GL_PROJECTION )
    gl.glLoadIdentity( )
    gl.glOrtho( 0, width, 0, height, -1, 1 )
    gl.glMatrixMode( gl.GL_MODELVIEW )
    gl.glLoadIdentity( )

def on_keyboard( key, x, y ):
    if key == '\033': sys.exit( )


if __name__ == '__main__':
    import sys
    glut.glutInit( sys.argv )
    glut.glutInitDisplayMode( glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH )
    glut.glutCreateWindow( "Freetype OpenGL" )
    glut.glutReshapeWindow( 600, 100 )
    glut.glutDisplayFunc( on_display )
    glut.glutReshapeFunc( on_reshape )
    glut.glutKeyboardFunc( on_keyboard )
    gl.glTexEnvf( gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE )
    gl.glEnable( gl.GL_DEPTH_TEST )
    gl.glEnable( gl.GL_BLEND )
    gl.glEnable( gl.GL_COLOR_MATERIAL )
    gl.glColorMaterial( gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE )
    gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
    gl.glEnable( gl.GL_TEXTURE_2D )
    makefont( '/Users/sac/Sites/runtime-2016-rev/pf_tempesta_seven.ttf', 16 )
    glut.glutMainLoop( )
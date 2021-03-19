"""scratch parts list"""


# example1.py
# play a sound to the left, to the right and to the center

# import the time standard module
import time

# import the pygame module
import pygame


# start pygame
pygame.init()

# load a sound file into memory
sound = pygame.mixer.Sound("bird.ogg")

# start playing the sound
# and remember on which channel it is being played
channel = sound.play()
# set the volume of the channel
# so the sound is only heard to the left
channel.set_volume(1, 0)
# wait for 1 second
time.sleep(1)

# do the same to the right
channel = sound.play()
channel.set_volume(0, 1)
time.sleep(1)

# do the same to the center
channel = sound.play()
channel.set_volume(1, 1)
time.sleep(1)



def checker():
    gc.collect()
    #global EntityHandler_tbk, Flag_tbk, Upgrade_tbk, PhysicalWorldElement_tbk
    for obj in gc.get_objects():
        mark = 0
        if isinstance(obj, EntityHandler):
            print('EntityHandler', obj, sys.getrefcount(obj))
            #print(gc.get_referrers(obj))
            del obj
            continue
            mark = 2

        if isinstance(obj, Flag):
            print('Flag', obj, sys.getrefcount(obj))
            #print(gc.get_referrers(obj))
            del obj
            continue
            mark = 2

        if isinstance(obj, Upgrade):
            print('Upgrade', obj, sys.getrefcount(obj))
            #print(gc.get_referrers(obj))
            del obj
            continue
            mark = 2

        if isinstance(obj, x29Element):
            if obj.name != "user":
                print('x29Element', obj, sys.getrefcount(obj))
                #print(gc.get_referrers(obj))
                del obj
                continue
                mark = 2

        if mark: del obj

    print "ckecked"





checker()





def getCleaner(class_arr, skip):
    for en in class_arr:
        if en.name != skip:
            class_arr.remove(en)
            print("getrefcount " + str(en), str(en.__class__), sys.getrefcount(en))
            del en






1: public
static

// Compute vectors
v0 = C - A
v1 = B - A
v2 = P - A

// Compute dot products
dot00 = dot(v0, v0)
dot01 = dot(v0, v1)
dot02 = dot(v0, v2)
dot11 = dot(v1, v1)
dot12 = dot(v1, v2)

// Compute barycentric coordinates
invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
u = (dot11 * dot02 - dot01 * dot12) * invDenom
v = (dot00 * dot12 - dot01 * dot02) * invDenom

// Check if point is in triangle
return (u >= 0) && (v >= 0) && (u + v < 1)

#m = [map(lambda y: x[0]+y, x[2]) for x in os.walk('audio/') if x[2][0] != '.']

def point_in_poly(P,vA,vB,vC):
    v0 = vC - vA
    v1 = vB - vA
    v2 = P - vA
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    #Compute barycentric coordinates
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    #Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)






#//OLD

dd = self._pos - elm._pos
if dd.length:
    collision = dd / dd.length
    bcf = aci = self.PHYS.Velo.dot(collision)
    acf = bci = elm.PHYS.Velo.dot(collision)

    self.PHYS.Velo += (acf - aci) * collision * 1.075
    elm.PHYS.Velo += (bcf - bci) * collision * 1.075

#//NEW
def collide(clsA,clsB):

    x = (clsB._pos - clsA._pos).normalize()

    x1d = x.dot(clsA.PHYS.Velo)
    v1x = x * x1d
    v1y = clsA.PHYS.Velo - v1x
    m1 = clsA.radius #mass

    x2d = -x.dot(clsB.PHYS.Velo) #Vector3.Dot(x, v2);
    v2x = -x * x2d
    v2y = clsB.PHYS.Velo - v2x
    m2 = clsB.radius #mass

    m1nm2 = m1 + m2

    clsA.PHYS.Velo = (v1x * ((m1 - m2) / m1nm2)) + (v2x * ((2 * m2) / m1nm2)) + v1y
    clsB.PHYS.Velo = (v1x * ((2 * m1) / m1nm2)) + (v2x * ((m2 - m1) / m1nm2)) + v2y




    x = (el._pos - self._pos).normalize()

    Vector3
    v1 = sphereA.Speed;

    float
    x1 = Vector3.Dot(x, v1);

    Vector3
    v1x = x * x1;
    Vector3
    v1y = v1 - v1x;

    float
    m1 = sphereA.Mass;
    x = -x;
    Vector3
    v2 = sphereB.Speed;
    float
    x2 = Vector3.Dot(x, v2);

    Vector3
    v2x = x * x2;
    Vector3
    v2y = v2 - v2x;

    float
    m2 = sphereB.Mass;
    float
    combinedMass = m1 + m2;

    Vector3
    newVelA = (v1x * ((m1 - m2) / combinedMass)) + (v2x * ((2 * m2) / combinedMass)) + v1y;
    Vector3
    newVelB = (v1x * ((2 * m1) / combinedMass)) + (v2x * ((m2 - m1) / combinedMass)) + v2y;

    sphereA.Speed = newVelA;
    sphereB.Speed = newVelB;

    #
# #ldp = Vector3(0.0,0.0,user_matrix.get_row_vec3(3).length)
# #npe = user_matrix.rotate_vec3(-ldp)
# #glCallList(draw_utility_line(Vector3(0.0,0.0,0.0), camera_matrix.get_row_vec3(3), camera_matrix.get_row_vec3(3)))
#
# # direc = user_matrix.get_row_vec3(3).unit().normalize() #Vector3(camera_matrix.forward)
# # u = user_matrix.get_row_vec3(3) + (direc * camera_distance)
# # camera_matrix.set_row(3, u.as_tuple())
# #print M
# # camera_matrix = camera_matrix.make_identity()
# # camera_matrix *= user_matrix
#
# au = user_matrix.get_row_vec3(3).unit().normalize()
# bu = camera_matrix.get_row_vec3(2).unit().normalize()
# v = au.dot(bu)#-(90.0*(pi/180.0))
# ax = bu.cross(au)
# #
# CAM_ATT_RAD = degrees(asin(v))
# #if abs(CAM_ATT_RAD) < 45.0:
#     #
#     # da = Vector3(0.0, 1.0, 0.0)
# M = Matrix44.rotation_about_axis(ax, v)
#
# ldp = Vector3(0.0,0.0,24.0)
#
# npe = M.rotate_vec3(-ldp)
#
# glCallList(draw_utility_line(user_matrix.get_row_vec3(3), ax*6, ax*-6))###user_matrix.get_row_vec3(3)))
#
#
#
# #camera_matrix *= M #camera_matrix.make_rotation_about_axis(ax, v)
#
#
# # da = Vector3(0.0, 1.0, 0.0)
# # ax = arv.cross(upos)



#pass





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


#
# dpos = user_matrix.get_row_vec3(3) #.normalize()#Vector3(camera_matrix.forward)
# upos = user_matrix.get_row_vec3(3).normalize()
#
# ppos = camera_matrix.get_row_vec3(3).normalize()
# #camera_matrix = camera_matrix.make_identity()
# arv = camera_matrix.get_row_vec3(2).normalize()
#
# v = arv.dot(ppos)  #-(90.0*(pi/180.0))
# CAM_ATT_RAD = degrees(asin(v))
# da = Vector3(0.0, 1.0, 0.0)
# ax = arv.cross(upos)
#
# # print camera_matrix
# # print "shite"
# # print user_matrix
# # print "user^"
# glCallList(draw_utility_line(dpos, arv, arv))
# glCallList(draw_utility_line(dpos, 4*ax, -4*ax))
# #
#
#
#
# direc = Vector3(camera_matrix.forward)
# u = dpos + (upos * camera_distance)

#camera_matrix.set_row(0, user_matrix.right)
#camera_matrix.set_row(1, user_matrix.up)
#camera_matrix.set_row(2, upos.as_tuple())






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

#
# cam_counter += 1
#
# da = Vector3(0.0, 1.0, 0.0)
# pve = Vector3(0.0, 0.0, 64.0)
# #;set_row(3, (0.0, 0.0, -64.0))
#
#
# #camera_matrix.make_rotation_about_axis(da, cam_counter * (pi / 180))
# npe = camera_matrix.rotate_vec3(pve)
#
# camera_matrix.translate = npe.as_tuple() #(0.0, 0.0, 128.0)
#



# R = ((bu.x * au.x, bu.x * au.y, bu.x * au.z, 0.0),
#      (bu.y * au.y, bu.y * au.x, bu.y * au.z, 0.0),
#      (bu.z * au.y, bu.z * au.x, bu.z * au.z, 0.0),
#      (0.0, 0.0, 0.0, 1.0))
# M = Matrix44()
# M.set(R[0], R[1], R[2], R[3])




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


pos = self._pos
fwd = Vector3(self.mat.forward) * -self.radius
u_gl_list = glGenLists(1)
glNewList(u_gl_list, GL_COMPILE)

glCallList(arrow)

glColor4f(0.5, 0.1, 0.1, 1.0)
glFrontFace(GL_CW)
glPushMatrix()
glTranslate(pos.x, pos.y, pos.z)
glMultMatrixf(self.mat.to_opengl())
#glutSolidIcosahedron()
glutSolidSphere(self.radius, 12, 12)
glPopMatrix()

glPushMatrix()
glTranslate(pos.x + fwd.x, pos.y + fwd.y, pos.z + fwd.z)
glutSolidSphere(self.radius * 0.2, 8, 8)
glPopMatrix()

glEndList()
return u_gl_list


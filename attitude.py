# module for rotation kinematics
import numpy as np

def rot1(angle):
    """
    Elementary rotation about the first axis. For row vectors b = a.dot(R)
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rot_mat = np.identity(3)
    rot_mat[1,1] = cos_a
    rot_mat[1,2] = sin_a
    rot_mat[2,1] = -sin_a
    rot_mat[2,2] = cos_a
    
    return rot_mat
    
def rot2(angle):
    """
    Elementary rotation about the second axis. For row vectors b = a.dot(R)
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rot_mat = np.identity(3)
    rot_mat[0,0] = cos_a
    rot_mat[0,2] = -sin_a
    rot_mat[2,0] = sin_a
    rot_mat[2,2] = cos_a
    
    return rot_mat
    
def rot3(angle):
    """
    Elementary rotation about the third axis. For row vectors b = a.dot(R)
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rot_mat = np.identity(3)
    rot_mat[0,0] = cos_a
    rot_mat[0,1] = sin_a
    rot_mat[1,0] = -sin_a
    rot_mat[1,1] = cos_a
    
    return rot_mat
 
def dcm2body313(dcm):
    """Convert DCM to body Euler 3-1-3 angles

    """
    theta = np.zeros(3)

    theta[0] = np.arctan2(dcm[2,0],dcm[2,1])
    theta[1] = np.arccos(dcm[2,2])
    theta[2]= np.arctan2(dcm[0,2],-dcm[1,2])

    return theta

def body313todcm(theta):
    dcm = np.zeros((3,3))
    st1 = np.sin(theta[0])
    st2 = np.sin(theta[1])
    st3 = np.sin(theta[2])
    ct1 = np.cos(theta[0])
    ct2 = np.cos(theta[1])
    ct3 = np.cos(theta[2])

    dcm[0,0] = ct1*ct3-ct2*st1*st3
    dcm[0,1] = -ct3*st1-ct1*ct2*st3
    dcm[0,2] = st3*st2
    dcm[1,0] = ct1*st3+ct3*ct2*st1
    dcm[1,1] = ct1*ct3*ct2-st1*st3
    dcm[1,2] = -ct3*st2
    dcm[2,0] = st1*st2
    dcm[2,1] = ct1*st2
    dcm[2,2] = ct2

    return dcm

def body313dot(theta,Omega):
    st1 = np.sin(theta[0])
    st2 = np.sin(theta[1])
    st3 = np.sin(theta[2])
    ct1 = np.cos(theta[0])
    ct2 = np.cos(theta[1])
    ct3 = np.cos(theta[2])

    mat = np.array([[-ct2*st1,-ct1*ct2, st2],
                    [st2*ct1, -st2*st1, 0],
                    [st1,ct1,0]])

    theta_dot = 1/st2*mat.dot(Omega)

    return theta_dot

def body313dot_to_ang_vel(theta,theta_dot):
    st1 = np.sin(theta[0])
    st2 = np.sin(theta[1])
    st3 = np.sin(theta[2])
    ct1 = np.cos(theta[0])
    ct2 = np.cos(theta[1])
    ct3 = np.cos(theta[2])

    mat = np.array([[0,ct1,st1*st2],
                    [0,-st1, ct1*st2],
                    [1,0,ct2]])

    Omega = mat.dot(theta_dot)
    return Omega

def hat_map(vec):
    """Return that hat map of a vector
    
    Inputs: 
        vec - 3 element vector

    Outputs:
        skew - 3,3 skew symmetric matrix

    """

    skew = np.array([
                    [0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])

    return skew

def vee_map(skew):
    """Return the vee map of a vector

    """

    vec = 1/2 * np.array([skew[2,1] - skew[1,2],
                          skew[0,2] - skew[2,0],
                          skew[1,0] - skew[0,1]])

    return vec

def dcmtoquat(dcm):
    """Convert DCM to quaternion
    
        Assume a positive scalar part of the quaternion

    """
    quat = np.zeros(4)
    quat[-1] = 1/2*np.sqrt(np.trace(dcm)+1)
    quat[0:3] = 1/4/quat[-1]*vee_map(dcm-dcm.T)

    return quat

def quattodcm(quat):
    """Convert quaternion to DCM

    Assume last element is the scalar part
    """

    dcm = (quat[-1]**2-np.inner(quat[0:3], quat[0:3]))*np.eye(3,3) + 2*np.outer(quat[0:3],quat[0:3]) + 2*quat[-1]*hat_map(quat[0:3])

    return dcm

def quatdot(quat, Omega):

    quat_dot = np.zeros(4)
    quat_dot[0:3] = 1/2*(quat[-1]*np.eye(3,3) + hat_map(quat[0:3])).dot(Omega)
    quat_dot[-1] = -1/2*Omega.dot(quat[0:3])

    return quat_dot

def quatdot_ang_vel(quat, quat_dot):
    q = quat[0:3]
    q4 = quat[-1]
    qd = quat_dot[0:3]
    q4d = quat_dot[-1]

    Omega = 2*(q4*qd - q4d*q - hat_map(q).dot(qd))

    return Omega

def dcmtoaxisangle(R):
    angle = np.arccos((np.trace(R) - 1)/2)
    axis = 1/(2*np.sin(angle))*vee_map(R-R.T)
        
    return angle, axis

def axisangletodcm(angle, axis):
    ahat = hat_map(axis)
    R = np.eye(3,3) + np.sin(angle)*ahat + (1 - np.cos(angle))*ahat.dot(ahat)
    return R

def dcmdottoang_vel(R,Rdot):
    """Convert a rotation matrix to angular velocity
        
        w - angular velocity in inertial frame
        Omega - angular velocity in body frame
    """
    w = vee_map(Rdot.dot(R.T))
    Omega = vee_map(R.T.dot(Rdot))

    return (w, Omega)

def ang_veltodcmdot(R,Omega):
    """Convert angular velocity to DCM dot
        Omega - angular velocity defined in body frame 
    """
    Rdot = R.dot(hat_map(Omega))

    return Rdot

def ang_veltoaxisangledot(angle, axis, Omega):
    """Compute kinematics for axis angle representation

    """
    angle_dot = axis.dot(Omega)
    axis_dot = 1/2*(hat_map(axis) - 1/np.tan(angle/2) * hat_map(axis).dot(hat_map(axis))).dot(Omega)
    return angle_dot, axis_dot

def axisangledottoang_vel(angle,axis, angle_dot,axis_dot):
    """Convert axis angle represetnation to angular velocity in body frame

    """
    
    Omega = angle_dot*axis + np.sin(angle)*axis_dot - (1-np.cos(angle))*hat_map(axis).dot(axis_dot)
    return Omega

def test_rot_mat_in_special_orthogonal_group(R):
    np.testing.assert_array_almost_equal(np.linalg.det(R), 1, decimal=2)
    np.testing.assert_array_almost_equal(R.T.dot(R), np.eye(3,3), decimal=2)

if __name__ == "__main__":
    angle = math.pi/4.0
    
    print(rot1(angle))
# COLLISIONS
import numpy as np
from dataclasses import dataclass
from bodies import Body, CollidableShape, AABB


@dataclass
class Contact:
    bodyA: Body
    bodyB: Body
    normal: np.ndarray                                      # Unit vector pointing from body A to body B
    depth: float                                            # The amount of penetration between bodies
    point: np.ndarray                                       # The point of collision 
    feature_id : int = 0                                    # persistant ID for warm start

@dataclass
class Endpoint:
    value: float                                            # the x-coordinate
    is_min: bool                                            # basically just says is this the start or the end of the AABB
    body: CollidableShape                                   # which body we are referencing


def broad_phase(bodies: list[CollidableShape], ignore_ids: set[tuple[int, int]]) ->list[tuple[CollidableShape, CollidableShape]]:
    """
    Find all potential colliding pairs using the Sweep and Prune algorithm.
    Speeds up calculation from nested for loops of O(n^2) to something like O(nlogn)
    """
    endpoints: list[Endpoint] = []                          # List of the start and end of the AABBs in the x-axis

    aabbs: dict[CollidableShape, AABB] = {body: body.get_aabb() for body in bodies} # Precaclulate aabbs
    for body in bodies:
        aabb = aabbs[body]
        endpoints.append(Endpoint(value=aabb.min_x, is_min=True, body=body))
        endpoints.append(Endpoint(value=aabb.max_x, is_min=False, body=body))

    endpoints.sort(key=lambda e:e.value)                    # Sort the endpoints list in order by their value only, not the other stuff

    active_list: list[CollidableShape] = []                 # Initialize active list - list of active bodies as we cycle through AABBs

    potential_pairs: set[tuple[int, int]] = set()           # Uses set because this handles duplicate pairs like (A, B) or (B, A)

    for endpoint in endpoints:                              # Sweep through the sorted list and form overlapping pairs, checks if they both contain same x value, then add to list if y vlaue also has overlap
        if endpoint.is_min:                                 # when we encounter a new body while cycling through sorted endpoints we then want to compare y values
            for other_body in active_list:                  # By definition these bpdies overlap on x
                aabb1 = endpoint.body.get_aabb()
                aabb2 = other_body.get_aabb()

                y_overlap = (aabb1.min_y <= aabb2.max_y) and (aabb2.min_y <= aabb1.max_y) # Check for overlap on y-axis

                if y_overlap:                               # If y overlaps and x already overlaps than the AABBs collide, add to detailed shape collision check
                    pair = tuple(sorted((id(endpoint.body), id(other_body))))
                    if ignore_ids and pair in ignore_ids:   # Don't add in ignored contacts
                        continue
                    potential_pairs.add(pair)

            active_list.append(endpoint.body)               # Add the current body we are inside to the active list - on first run goes right to this to start building the active list

        else:                                               # The else is when we leave the domain of the current AABB - remove from active list
            try:                                            # Try and Except are just safer ways to remove stuff, all it is really
                active_list.remove(endpoint.body)
            except ValueError:
                pass

    body_map = {id(b): b for b in bodies}                   # Somehow this maps the ids in the potential_pairs set back to the bodies
    return [(body_map[id1], body_map[id2]) for id1, id2 in potential_pairs] 


def project_on_axis(body: CollidableShape, axis: np.ndarray) -> tuple[float, float]:
    """
    Helper function for the SAT. Project the corners of a body onto an axis.
    """

    corners = body.get_corners()
    projections = corners @ axis                            # Matrix multiplication to project corner vectors onto the axis

    return np.min(projections), np.max(projections)


def find_contact_point(bodyA: CollidableShape, bodyB: CollidableShape, normal: np.ndarray) -> np.ndarray:
    """
    Find the vertex on body A that is is most deeply penetrating body B
    """
    cornersA = bodyA.get_corners()

    projections = cornersA @ normal                         # Project all corners onto the normal vector of penetration, the "leading" corner is the penetration

    deepest_corner_index = np.argmax(projections)           # This is the corner that penetrates deepest
    
    return cornersA[deepest_corner_index]                   # Return the world coordinates of the corner

    
def narrow_phase(pair: tuple[CollidableShape, CollidableShape]):
    """
    Performs the Separating Axis Test (SAT) on a pair of bodies.
    Project a the "shadows" of two bodies onto an axis to see if the shadows overlap (colliding if so).
    Trying to find just one axis that proves separation - if so we can definitively say no collision.
    """

    bodyA, bodyB = pair

    if bodyA.static and not bodyB.static:
        bodyA, bodyB = bodyB, bodyA                         # Swap bodies if A is static - mathematical convention ?
    if bodyA.static and bodyB.static:
        return None                                         # No contacts between the two static, unmoveable objects

    min_overlap = float('inf')
    collision_normal = None

    axes_A = bodyA.get_axes()                               # Get the axis that are perpendicular to each of the body's faces
    axes_B = bodyB.get_axes()

    all_axes = np.vstack([axes_A, axes_B])

    for axis in all_axes:                                   # Project both bodies onto the current axis - get their shadows
        minA, maxA = project_on_axis(bodyA, axis)           # minA and maxA are the corners projected onto the given axis
        minB, maxB = project_on_axis(bodyB, axis)

        if maxA <= minB or maxB <= minA:                    # Check for no overlap - if no overlap exit 
            return None
        
        overlap = min(maxA, maxB) - max(minA, minB)         # If overlap than calculate how much overlap

        if overlap < min_overlap:                           # Save smallest overlap - most efficient place to push bodies apart is axis of smallest overlap
            min_overlap = overlap
            collision_normal = axis

    center_direction = bodyB.position[:2] - bodyA.position[:2]
    if np.dot(center_direction, collision_normal) < 0:
        collision_normal = -collision_normal

    contact_point = find_contact_point(bodyA, bodyB, collision_normal)

    return Contact(bodyA, bodyB, normal=collision_normal, depth=min_overlap, point=contact_point)


def get_collisions(bodies: list[Body], ignore_ids: set[tuple[int, int]] | None = None) -> list[Contact]:
    """
    Pass in a list of bodies and detect_collision will return a list of colliding bodies.
    Runs the broad & narrow phases of collision detection.
    """

    all_contacts: list[Contact]  = []

    colliable_bodies = [b for b in bodies if isinstance(b, CollidableShape)]

    # Phase 1: Broad Phase
    potential_pairs = broad_phase(colliable_bodies, ignore_ids=ignore_ids)

    # Phase 2: Narrow Phase
    for pair in potential_pairs:
        contact_info = narrow_phase(pair)
        if contact_info:
            all_contacts.append(contact_info)
    return all_contacts
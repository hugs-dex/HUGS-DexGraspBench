import torch
import pytorch_kinematics as pk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import difflib
import argparse


def compare_joint_names(urdf_chain, mjcf_chain):
    """
    Comprehensive joint name comparison between URDF and MJCF chains
    """
    # Get joint information
    urdf_joints = [(joint.name, joint.joint_type) for joint in urdf_chain.get_joints()]
    mjcf_joints = [(joint.name, joint.joint_type) for joint in mjcf_chain.get_joints()]

    urdf_joint_names = [joint[0] for joint in urdf_joints]
    mjcf_joint_names = [joint[0] for joint in mjcf_joints]

    urdf_joint_types = {joint[0]: joint[1] for joint in urdf_joints}
    mjcf_joint_types = {joint[0]: joint[1] for joint in mjcf_joints}

    print("=" * 60)
    print("JOINT NAME COMPARISON")
    print("=" * 60)

    # Basic counts
    print(f"URDF joints: {len(urdf_joint_names)}")
    print(f"MJCF joints: {len(mjcf_joint_names)}")
    print()

    # Find matches and differences
    common_joints = set(urdf_joint_names) & set(mjcf_joint_names)
    urdf_only = set(urdf_joint_names) - set(mjcf_joint_names)
    mjcf_only = set(mjcf_joint_names) - set(urdf_joint_names)

    print(f"Common joints: {len(common_joints)}")
    print(f"Joints only in URDF: {len(urdf_only)}")
    print(f"Joints only in MJCF: {len(mjcf_only)}")
    print()

    # Detailed analysis
    if common_joints:
        print("COMMON JOINTS:")
        for joint in sorted(common_joints):
            urdf_type = urdf_joint_types[joint]
            mjcf_type = mjcf_joint_types[joint]
            type_match = "✅" if urdf_type == mjcf_type else "❌"
            print(f"  {joint}: URDF({urdf_type}) {type_match} MJCF({mjcf_type})")
        print()

    if urdf_only:
        print("JOINTS ONLY IN URDF:")
        for joint in sorted(urdf_only):
            print(f"  {joint} ({urdf_joint_types[joint]})")

        # Find similar names in MJCF
        print("\n  Closest matches in MJCF:")
        for urdf_joint in sorted(urdf_only):
            matches = difflib.get_close_matches(urdf_joint, mjcf_joint_names, n=3, cutoff=0.6)
            if matches:
                print(f"    {urdf_joint} -> {matches}")
        print()

    if mjcf_only:
        print("JOINTS ONLY IN MJCF:")
        for joint in sorted(mjcf_only):
            print(f"  {joint} ({mjcf_joint_types[joint]})")

        # Find similar names in URDF
        print("\n  Closest matches in URDF:")
        for mjcf_joint in sorted(mjcf_only):
            matches = difflib.get_close_matches(mjcf_joint, urdf_joint_names, n=3, cutoff=0.6)
            if matches:
                print(f"    {mjcf_joint} -> {matches}")
        print()

    # Joint type consistency check
    type_mismatches = []
    for joint in common_joints:
        if urdf_joint_types[joint] != mjcf_joint_types[joint]:
            type_mismatches.append((joint, urdf_joint_types[joint], mjcf_joint_types[joint]))

    if type_mismatches:
        print("JOINT TYPE MISMATCHES:")
        for joint, urdf_type, mjcf_type in type_mismatches:
            print(f"  {joint}: URDF={urdf_type} vs MJCF={mjcf_type}")
        print()

    return {
        "common_joints": sorted(common_joints),
        "urdf_only": sorted(urdf_only),
        "mjcf_only": sorted(mjcf_only),
        "type_mismatches": type_mismatches,
    }


def validate_joint_hierarchy(urdf_chain, mjcf_chain, common_joints):
    """
    Validate that common joints have the same hierarchical structure
    """
    print("JOINT HIERARCHY VALIDATION")
    print("=" * 40)

    # Build parent-child relationships
    def build_hierarchy(chain):
        hierarchy = {}
        for joint in chain.get_joints():
            hierarchy[joint.name] = {"parent": getattr(joint, "parent", None), "child": getattr(joint, "child", None)}
        return hierarchy

    urdf_hierarchy = build_hierarchy(urdf_chain)
    mjcf_hierarchy = build_hierarchy(mjcf_chain)

    hierarchy_issues = []

    for joint in common_joints:
        urdf_parent = urdf_hierarchy[joint].get("parent", "Unknown")
        mjcf_parent = mjcf_hierarchy[joint].get("parent", "Unknown")

        if urdf_parent != mjcf_parent:
            hierarchy_issues.append((joint, urdf_parent, mjcf_parent))
            print(f"❌ {joint}: URDF parent='{urdf_parent}' vs MJCF parent='{mjcf_parent}'")
        else:
            print(f"✅ {joint}: parent='{urdf_parent}'")

    if not hierarchy_issues:
        print("All common joints have consistent hierarchy!")
    else:
        print(f"\nFound {len(hierarchy_issues)} hierarchy mismatches")

    return hierarchy_issues


def test_fk_consistency(urdf_chain, mjcf_chain, common_joints, num_samples=1000, max_joint_angle=3.14):
    """
    Test FK consistency between URDF and MJCF chains for common joints
    """
    device = urdf_chain.device
    dtype = urdf_chain.dtype

    # Get all joint names
    urdf_joint_names = [joint.name for joint in urdf_chain.get_joints()]
    mjcf_joint_names = [joint.name for joint in mjcf_chain.get_joints()]

    print("\nFK CONSISTENCY TESTING")
    print("=" * 40)
    print(f"Testing {num_samples} random configurations")
    print(f"Common joints used: {common_joints}")

    # Generate random joint configurations for common joints
    batch_size = 100
    errors = []

    with tqdm(total=num_samples, desc="Testing FK consistency") as pbar:
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)

            # Generate random joint angles for common joints
            theta_common = (
                torch.rand((current_batch, len(common_joints)), device=device, dtype=dtype) * 2 * max_joint_angle
                - max_joint_angle
            )

            # Create full joint configurations for both chains
            theta_urdf = torch.zeros((current_batch, len(urdf_joint_names)), device=device, dtype=dtype)
            theta_mjcf = torch.zeros((current_batch, len(mjcf_joint_names)), device=device, dtype=dtype)

            # Map common joints to their respective positions
            for j, joint_name in enumerate(common_joints):
                urdf_idx = urdf_joint_names.index(joint_name)
                mjcf_idx = mjcf_joint_names.index(joint_name)
                theta_urdf[:, urdf_idx] = theta_common[:, j]
                theta_mjcf[:, mjcf_idx] = theta_common[:, j]

            # Forward kinematics for both chains
            with torch.no_grad():
                urdf_transforms = urdf_chain.forward_kinematics(theta_urdf)
                mjcf_transforms = mjcf_chain.forward_kinematics(theta_mjcf)

            # Compare end-effector poses for all common links
            common_links = set(urdf_transforms.keys()) & set(mjcf_transforms.keys())

            for link_name in common_links:
                # Get transformation matrices
                T_urdf = urdf_transforms[link_name].get_matrix()
                T_mjcf = mjcf_transforms[link_name].get_matrix()

                # Calculate position error
                pos_urdf = T_urdf[:, :3, 3]
                pos_mjcf = T_mjcf[:, :3, 3]
                pos_error = torch.norm(pos_urdf - pos_mjcf, dim=1)

                # Calculate orientation error
                R_urdf = T_urdf[:, :3, :3]
                R_mjcf = T_mjcf[:, :3, :3]
                R_diff = torch.bmm(R_urdf, R_mjcf.transpose(1, 2))

                trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
                angle_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

                # Store errors
                for batch_idx in range(current_batch):
                    errors.append(
                        {
                            "link": link_name,
                            "position_error": pos_error[batch_idx].item(),
                            "orientation_error": angle_error[batch_idx].item(),
                            "config_idx": i + batch_idx,
                        }
                    )

            pbar.update(current_batch)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check consistency between URDF and MJCF files")
    parser.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to MJCF file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    print(f"Using device: {device}, dtype: {dtype}")

    # Load chains
    print("Loading URDF chain...")
    urdf_chain = pk.build_chain_from_urdf(open(args.urdf).read()).to(device=device, dtype=dtype)

    print("Loading MJCF chain...")
    mjcf_chain = pk.build_chain_from_mjcf(args.mjcf).to(device=device, dtype=dtype)

    # Step 1: Comprehensive joint name comparison
    joint_analysis = compare_joint_names(urdf_chain, mjcf_chain)

    # Step 2: Validate joint hierarchy
    hierarchy_issues = validate_joint_hierarchy(urdf_chain, mjcf_chain, joint_analysis["common_joints"])

    # Only proceed with FK testing if we have common joints and no critical issues
    if not joint_analysis["common_joints"]:
        print("\n❌ No common joints found! Cannot perform FK consistency testing.")
        return

    if joint_analysis["type_mismatches"]:
        print("\n⚠️  Joint type mismatches detected. FK results may be inconsistent.")

    if hierarchy_issues:
        print("\n⚠️  Joint hierarchy mismatches detected. FK results may be inconsistent.")

    # Step 3: FK consistency testing
    errors = test_fk_consistency(
        urdf_chain,
        mjcf_chain,
        joint_analysis["common_joints"],
        num_samples=500,  # Reduced for demonstration
        max_joint_angle=3.14,
    )

    # Analyze results
    if errors:
        pos_errors = np.array([e["position_error"] for e in errors])
        orient_errors = np.array([e["orientation_error"] for e in errors])

        print("\nFK CONSISTENCY RESULTS:")
        print(f"Position error - Mean: {pos_errors.mean():.6f}, Max: {pos_errors.max():.6f}")
        print(f"Orientation error - Mean: {orient_errors.mean():.6f}, Max: {orient_errors.max():.6f}")

        # Consistency check
        tolerance = 1e-6
        if pos_errors.max() < tolerance and orient_errors.max() < tolerance:
            print("✅ FK results are CONSISTENT within tolerance!")
        else:
            print("❌ FK results are INCONSISTENT!")

        # Group errors by link name
        link_errors = {}
        for error in errors:
            link_name = error["link"]
            if link_name not in link_errors:
                link_errors[link_name] = {"position_errors": [], "orientation_errors": []}
            link_errors[link_name]["position_errors"].append(error["position_error"])
            link_errors[link_name]["orientation_errors"].append(error["orientation_error"])

        # Print aggregated results by link
        print("\nInconsistant Links:")
        print("=" * 80)
        for link_name in sorted(link_errors.keys()):
            pos_err = np.array(link_errors[link_name]["position_errors"])
            orient_err = np.array(link_errors[link_name]["orientation_errors"])
            if pos_err.max() > 1e-3 or orient_err.max() > 1e-3:
                print(f"\nLink: {link_name}")
                print(
                    f"  Position error    - Mean: {pos_err.mean():.8f}, Max: {pos_err.max():.8f}, Min: {pos_err.min():.8f}, Std: {pos_err.std():.8f}"
                )
                print(
                    f"  Orientation error - Mean: {orient_err.mean():.8f}, Max: {orient_err.max():.8f}, Min: {orient_err.min():.8f}, Std: {orient_err.std():.8f}"
                )
                print(f"  Samples: {len(pos_err)}")

    else:
        print("No FK errors calculated - check if common links exist.")


if __name__ == "__main__":
    main()

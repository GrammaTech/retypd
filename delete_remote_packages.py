#!/usr/bin/env python
"""
Helper script for working around a "feature" in Gitlab's pypi package repositories.
Pypi does not support uploading packages of the same version,
they believe the version should be bumped for any change.
We're going to delete packages instead.

Call this script with positional args.
The first is the token with read and write access to the pypi repository.
This can be your personal access token or $CI_DEPLOY_PASSWORD.
$CI_JOB_TOKEN does not work, it doesn't have write access.
The remainder is an unlimited list of wheels to be uploaded.

If any local wheels find a match in name and version in the remote pypi repository, the remote will be deleted.
"""
import sys
import requests
import pkginfo

TOKEN = sys.argv[1]
for wheel_loc in sys.argv[2:]:
    wheel = pkginfo.Wheel(wheel_loc)
    # Get packages, handling pagination
    responses = []
    response = requests.get(
        f'https://git.grammatech.com/api/v4/projects/1587/packages?package_name={wheel.name}',
        headers={'PRIVATE-TOKEN': TOKEN},
    )
    responses.append(response)
    while response.links.get('next'):
        response = requests.get(response.links.get('next')['url'],
                                headers={'PRIVATE-TOKEN': TOKEN})
        if response.status_code != 200:
            raise Exception(f'{response.status_code} status code while requesting package listings filtered by local name: {wheel.name}')
        responses.append(response)

    packages = [package for response in responses for package in response.json()]
    # Delete all matching packages
    for package in packages:
        if wheel.version == package['version'] and wheel.name == package['name']:
            print(f'Deleting {package["name"]} {package["version"]}.')
            response = requests.delete(
                f'https://git.grammatech.com/api/v4/projects/1587/packages/{package["id"]}',
                headers={'PRIVATE-TOKEN': TOKEN},
            )
            if response.status_code != 204:
                raise Exception(f'{response.status_code} status code while deleting this package: {package["name"]} {package["version"]}')

"""Build or deploy a static web scene of land and building values.

Examples
--------
Build the Boston pilot into a local folder, then upload it::

    python -m openplaces.viz.web_scene build --region boston-inner-core \
        --admin_ids US-MA-BOS US-MA-CAM US-MA-SOM US-MA-BRE \
        --output_dir C:/earth/share/web/boston-values
    python -m openplaces.viz.web_scene deploy \
        --output_dir C:/earth/share/web/boston-values --go

The parcel and footprint delivery bundles of the region must exist
(`export_delivery`). The deploy reads the host, user, folder and key
from `OPENPLACES_WEB_DEPLOY_*` variables or `web-deploy.env` in the
openplaces config folder (`load_deploy_settings`); without `--go` it
is a dry run.
"""

import argparse

from openplaces.io.readers import get_region_admin_ids
from openplaces.viz.web_scene import build_web_scene, deploy_web_scene


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m openplaces.viz.web_scene')
    commands = parser.add_subparsers(dest='command', required=True)

    build = commands.add_parser('build', help='Write the site into a local folder.')
    build.add_argument(
        '--region',
        required=True,
        nargs='+',
        help='Delivery region id(s); each unit is read from its own region.',
    )
    build.add_argument(
        '--admin_ids',
        nargs='*',
        default=None,
        help='Units of the region to include (default: all of them).',
    )
    build.add_argument('--output_dir', required=True)
    build.add_argument('--title', default=None)
    build.add_argument('--build_id', default=None)
    build.add_argument(
        '--elevation_recipe',
        default=None,
        help='DEM recipe for 3D terrain, e.g. US_land-elevation-usgs-3dep.',
    )
    build.add_argument('--terrain_exaggeration', type=float, default=1.0)

    deploy = commands.add_parser(
        'deploy', help='Upload a built site (a dry run unless --go).'
    )
    deploy.add_argument('--output_dir', required=True)
    deploy.add_argument(
        '--env_file',
        default=None,
        help='Deploy settings file (default: web-deploy.env in the openplaces '
        'config folder, beside credentials.yaml).',
    )
    deploy.add_argument('--go', action='store_true', help='Upload for real.')
    deploy.add_argument(
        '--subpath',
        default=None,
        help="Folder under the configured path, e.g. a place's slug.",
    )

    args = parser.parse_args(argv)
    if args.command == 'build':
        admin_ids = args.admin_ids or [
            a for r in args.region for a in get_region_admin_ids(r)
        ]
        build_web_scene(
            args.region if len(args.region) > 1 else args.region[0],
            admin_ids,
            args.output_dir,
            title=args.title,
            build_id=args.build_id,
            elevation_recipe=args.elevation_recipe,
            terrain_exaggeration=args.terrain_exaggeration,
        )
    else:
        # Make overwriting the public site deliberate. CI pipes the
        # answer in, having already decided to publish.
        if args.go:
            answer = input('This overwrites the LIVE site. Type "live" to go: ')
            if answer.strip() != 'live':
                print('Cancelled.')
                return
        deploy_web_scene(
            args.output_dir, env_file=args.env_file, go=args.go, subpath=args.subpath
        )


if __name__ == '__main__':
    main()

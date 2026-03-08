{ pkgs, lib, config, ... }:

{
  dotenv.enable = true;

  packages = [ pkgs.zlib ];

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  enterShell = ''
    echo "$(python --version | cut -d' ' -f2), uv $(uv --version | cut -d' ' -f2)"
  '';
}

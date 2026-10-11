{
  description = "Qwen3.8-Flash-Next on an 8GB NVIDIA GPU, plus the isolated dev container";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      lib = nixpkgs.lib;

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      cuda = pkgs.cudaPackages_12_8.cudatoolkit;

      # =====================================================================
      # Model configuration -- serve and bench MUST agree.
      #
      # -cs is the KV cache in tokens, and it is the conversation ceiling.
      #
      # Sizing it: this model keeps only 12 of its 48 layers in a real KV cache
      # (the other 36 are linear attention with fixed-size recurrent state), and
      # those 12 use 2 KV heads at head_dim 256 -- but the QSA indexer adds side
      # planes, so it measures ~27.8 KiB/token, not the 24 a back-of-envelope
      # gives you. The startup "VRAM / KV cache" line prints the real figure.
      #
      # IMPORTANT: you cannot size this from the startup report's free-VRAM
      # number. Loading transiently needs several hundred MiB more than steady
      # state (autosplit runs a measuring forward pass), so a cache that leaves
      # 400 MiB free at runtime can still fail to LOAD:
      #
      #     -cs 16384   loads.  KV 444 MiB, 842 MiB free at runtime.
      #     -cs 32768   Runtime math said ~398 MiB free. It does not even load:
      #                 "Insufficient VRAM in split for model and cache".
      #
      # The real ceiling is somewhere between those. Raise this only after
      # running the probe below -- each attempt is one model load.
      #
      # -cs does NOT change decode speed. Measured on this box, -cs 4096 gave
      # 3.82 tok/s at 262.0 ms/token and -cs 20480 gave 3.81 at 262.2 -- identical.
      # Size it for the conversation you want, not for throughput.
      # =====================================================================
      contextSize = 20480;

      # MoE layers whose routed experts run on the CPU. This is the flag that
      # makes a 177B model fit in 8 GB at all; without it ExLlamaV3 wants ~52 GB.
      moeCpuOffload = 64;

      # Sizes the probe walks upward, stopping at the first that loads.
      probeSizes = [ 20480 24576 28672 32768 ];

      # Pin -mct here to override the automatic choice below.
      cpuThreads = null;

      modelFlags = lib.concatStringsSep " " (
        [ "-mcl" (toString moeCpuOffload) "-cs" (toString contextSize) ]
        ++ lib.optionals (cpuThreads != null) [ "-mct" (toString cpuThreads) ]
      );

      # Worker threads for the CPU MoE, resolved at RUN time.
      #
      # ExLlamaV3's own default is cpu_count/2, which is the wrong number on an
      # SMT machine. Measured here on 24 physical / 48 logical cores:
      #
      #     16 threads  (its default)  3.78 tok/s at ~50% CPU
      #     24 threads  (1 per core)   4.48 tok/s at  80% CPU
      #     32 threads                 2.69 tok/s
      #     48 threads  (nproc)        1.90 tok/s
      #
      # Two SMT siblings per core halve throughput: the CPU MoE is already
      # bandwidth-starved, so oversubscribing contends for the same memory
      # controllers. One thread per PHYSICAL core is the setting to use.
      #
      # QWEN_MOE_THREADS overrides at run time, e.g. QWEN_MOE_THREADS="-mct 24".
      threadsArg = lib.optionalString (cpuThreads == null) ''
        phys=$(awk -F: '/^physical id/{p=$2} /^core id/{print p"-"$2}' /proc/cpuinfo \
               | sort -u | wc -l)
        [ "$phys" -gt 0 ] || phys=$(nproc)
        read -r -a moeThreads <<< "''${QWEN_MOE_THREADS:--mct $phys}"
      '' + lib.optionalString (cpuThreads != null) ''
        moeThreads=()
      '';

      # Everything up to (but not including) the final exec.
      setup = ''
        export CUDA_HOME="${cuda}"
        export CUDA_PATH="$CUDA_HOME"
        export LD_LIBRARY_PATH="${cuda}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export PATH="${cuda}/bin:$PATH"

        ROOT="''${XDG_DATA_HOME:-$HOME/.local/share}/qwen38"
        MODEL="$ROOT/model"
        VENV="$ROOT/venv"
        REPO="$ROOT/runner"

        mkdir -p "$ROOT"

        # Run the tree this flake was evaluated from, NOT a fresh clone of
        # GitHub. The clone version meant `nix run .#bench` silently tested
        # origin/main and ignored the working tree, so an edit could not change
        # the result until it was committed and pushed.
        #
        # NOTE: `nix run .` evaluates git HEAD. To test uncommitted edits, use
        # `nix run path:.#bench` -- that is what the "Git tree is dirty" warning
        # is telling you.
        rm -rf "$REPO"
        cp -r ${self} "$REPO"
        chmod -R u+w "$REPO"

        python -m venv "$VENV"

        "$VENV/bin/python" -m pip install --upgrade pip
        "$VENV/bin/python" -m pip install torch \
          --index-url https://download.pytorch.org/whl/cu128
        "$VENV/bin/python" -m pip install exllamav3==1.4.5 huggingface_hub

        if [ ! -f "$MODEL/.download-complete" ]; then
          until HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=300 \
            "$VENV/bin/hf" download turboderp/Qwen3.8-Flash-Next-exl3 \
              --revision 3.05bpw_h5_ng5 \
              --local-dir "$MODEL"; do
            echo "Download interrupted; retrying in 15 seconds..."
            sleep 15
          done
          touch "$MODEL/.download-complete"
        fi

        cd "$REPO"
      '';

      runtimeInputs = [
        pkgs.python3
        pkgs.coreutils
        pkgs.gawk
        pkgs.curl
        pkgs.gnumake
        pkgs.ninja
        pkgs.stdenv.cc
        cuda
      ];

      serve = pkgs.writeShellApplication {
        name = "qwen38";
        inherit runtimeInputs;
        text = setup + threadsArg + ''
          exec "$VENV/bin/python" serve.py -m "$MODEL" ${modelFlags} "''${moeThreads[@]}" \
            --port 8080 --host 0.0.0.0 "$@"
        '';
      };

      # Same model configuration as serve, so the numbers describe what you
      # actually run. The README's published table was measured at -cs 4096; to
      # reproduce that exactly, override:  nix run .#bench -- -cs 4096
      # Extra args after `--` are passed through.
      bench = pkgs.writeShellApplication {
        name = "qwen38-bench";
        inherit runtimeInputs;
        text = setup + threadsArg + ''
          exec "$VENV/bin/python" bench.py -m "$MODEL" ${modelFlags} "''${moeThreads[@]}" "$@"
        '';
      };

      # Find the largest -cs that actually loads. Each step is one full model
      # load, so budget a few minutes. Use this instead of reasoning about
      # free VRAM at runtime: the two differ by the loading transient, and the
      # difference is large enough to turn a "looks fine" number into a crash.
      probe = pkgs.writeShellApplication {
        name = "qwen38-probe";
        inherit runtimeInputs;
        text = setup + ''
          for cs in ${lib.concatStringsSep " " probeSizes}; do
            echo "=== -cs $cs"
            if out=$("$VENV/bin/python" bench.py -m "$MODEL" -mcl ${toString moeCpuOffload} \
                       --ntok 4 -cs "$cs" 2>&1); then
              echo "    FITS:"
              grep -E "KV cache" <<<"$out" | sed "s/^/      /"
              echo "    -> set contextSize = $cs in flake.nix, then:"
              echo "       opencode.json: limit.context <= $((cs - 4096))"
              break
            else
              echo "    too big (does not load)"
            fi
          done
        '';
      };

      # =====================================================================
      # Isolated dev container -- nix run .#isolated-build
      #                    nix run .#isolated-nobuild
      # =====================================================================
      commonBuildInputs = [ pkgs.git pkgs.stdenv.cc ];

      greet = ''
        echo "===================================="
        echo " Welcome to the deterministic dev shell! "
        echo "===================================="
      '';

      policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
      containername = "qwen-isolated-dev";

      podmanRuntimePath = lib.makeBinPath [ pkgs.slirp4netns ];

      podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
        + "--network=slirp4netns "
        + "--tmpfs /tmp "
        + "-v .:/workspace/qwen:z "
        + "-e HOME=/root "
        + "${containername}:latest /bin/entrypoint.sh";

      isolatedImage = pkgs.dockerTools.buildImage {
        name = containername;
        tag = "latest";
        copyToRoot = pkgs.buildEnv {
          name = containername;
          paths = commonBuildInputs ++ [
            pkgs.bashInteractive
            pkgs.ripgrep
            pkgs.git
            pkgs.opencode
            pkgs.busybox
            (pkgs.writeScriptBin "entrypoint.sh" ''
              #!${pkgs.bashInteractive}/bin/bash
              ${greet}
              exec ${pkgs.bashInteractive}/bin/bash
            '')
          ];
          pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
        };
        config = {
          Env = [ "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt" "HOME=/root" ];
          Cmd = [ "/bin/entrypoint.sh" ];
          WorkingDir = "/workspace/qwen";
        };
      };
    in {
      packages.${system} = {
        default = serve;
        inherit serve bench probe isolatedImage;
      };

      apps.${system} = {
        default = { type = "app"; program = "${serve}/bin/qwen38"; };
        serve = { type = "app"; program = "${serve}/bin/qwen38"; };
        bench = { type = "app"; program = "${bench}/bin/qwen38-bench"; };
        probe = { type = "app"; program = "${probe}/bin/qwen38-probe"; };

        isolated-build = {
          type = "app";
          program = lib.getExe' pkgs.writeShellApplication {
            name = containername;
            runtimeInputs = [ pkgs.podman pkgs.slirp4netns ];
            text = ''
              export PATH="${podmanRuntimePath}:$PATH"
              ${pkgs.podman}/bin/podman rmi ${containername} || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${isolatedImage}
              ${podmanRun}
            '';
          };
        };

        isolated-nobuild = {
          type = "app";
          program = lib.getExe' pkgs.writeShellApplication {
            name = "run-isolated";
            runtimeInputs = [ pkgs.podman pkgs.slirp4netns ];
            text = ''
              export PATH="${podmanRuntimePath}:$PATH"
              ${podmanRun}
            '';
          };
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = commonBuildInputs;
        shellHook = greet;
      };
    };
}

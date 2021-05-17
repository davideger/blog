# Mourning Moore, or: Is that sexy RTX 3090 really worth [$3000](https://offer.ebay.com/ws/eBayISAPI.dll?ViewBidsLogin&item=254885216032)?

May 2021

David Eger

I’ve been playing around [fast.ai](https://www.fast.ai/) lately and
training image recognition models for different kinds of birds based
on labeled photos from Cornell's [Macaulay Library](https://www.macaulaylibrary.org/).
Thanks to crazy python compiler tricks like [JAX](https://github.com/google/jax),
[pytorch](https://pytorch.org/) and Nvidia's [CUDA](https://developer.nvidia.com/cuda-toolkit),
you can write codes in beautifully short Python and they gets
jit’d into massively parallel programs that take advantage of thousands
of GPU, CPU, or TPU cores and blindingly fast matrix multiplies.

My computer is about 8 years old, from a time when good consumer CPUs had 4 cores,
and long before those instantaneous matrix multiplying tensor cores were a
twinkle in Jensen Huang’s eye.

So when my machine decided to die a month ago, I wasn’t that
distraught.  My 4 CPU cores had been completely pegged during
ML training, and the death rattle of my elderly PC was a great excuse
to get a top-of-the-line ML workstation.  Everything, and I mean
everything, had *supposedly* much faster.

Lisa “Su is for Superwoman” launched Zen3 in Fall 2020 so there
were hot new sub-$1k AMD chips that scored
**[8x](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+9+5900X&id=3870)**
the cpumark of my now [geriatric Intel
CPU](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-3570K+%40+3.40GHz&id=828).
The new NVMe M.2 drives boasted **6.3x** the read speed of my ol’
SATA III SSD.  The Santa Clara sensation’s RTX 3090 now sported
**4x** the Cuda cores, shiny new tensor cores that do matrix
multiplies **9x faster**, and have **3x** the RAM of my now middle
aged GTX 1080.  And all that delicious training data could be
shoveled from host memory to GPU and back at twice the speed on
those hot new 16xPCIe 4.0 lanes.

Averaged together, I figured "a new top of the line should speed
my training up from 5 minutes per epoch to 1 minute per epoch!"
That sort of boost might be worth plonking down $3.5k on a new machine:
if ML training takes 12 minutes instead of an hour, that means you
can try a dozen ideas in an afternoon instead of just three as you're figuring
out what might work.

Unfortunately, those latest-and-greatest parts, especially the GPUs
were incredibly difficult to *get*, often simply unavailable or
fetching more than 2x their MSRP in the [secondary
markets](https://offer.ebay.com/ws/eBayISAPI.dll?ViewBidsLogin&item=254885216032).
So I ordered a pre-built one of the few ways you can actually get an RTX 3090.
When I got it, I loaded up the monster
with [Ubuntu 21.04](https://ubuntu.com/blog/ubuntu-21-04-is-here)
straight off the presses, the bleeding edge [CUDA
11](https://developer.nvidia.com/cuda-toolkit), and strapped in to
find... approximately one iteration of Moore’s law in about a decade.

Comparing this high end PC built in April 2021 to a PC put together
for half the cost in ~2017, total training time for my birdie
image classifier went from about 62 minutes to 25 minutes.

## Old Rig
+ GPU: Gigabyte GTX 1080 (8GB RAM / launched 2017)
+ CPU: Core i5 3570K (4 core / launched 2012) - [cpumark of 4,921](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-3570K+%40+3.40GHz&id=828)

## New Rig
+ GPU: Geforce RTX 3090 (24GB RAM / launched 2020)
+ CPU: Ryzen 5900 (12 core / launched 2020, AMD) - [cpumark of 39,478](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+9+5900X&id=3870)

## Fastai CNN training (resnet34 over 67k images)

Build Year | Build | batch size | CNN Training+Validation one epoch (seconds) | as MM:SS | speedup | speedup per year  
-----------|-------|-----------------------------------------|-------|---------|---------------|------
2017 | Core i5 3570K (‘12) + GTX 1080 (‘16) | 128 | 310 seconds | 5:10 | 1.0 |
2021 | Ryzen 5900 (‘20) + GTX 1080 (‘16)    | 128 | 189 seconds | 3:09 | 1.6x | 1.08x
2021 | Ryzen 5900 (‘20) + RTX 3090 (‘20)    | 128 | 127 seconds | 2:07 | 2.4x | 1.16x
2021 | Ryzen 5900 (‘20) + RTX 3090 (‘20)    | 350 | 119 seconds | 1:59 | 2.6x | 1.17x

The brand new high-end machine *is* faster, but for a coder of
my vintage the speed up is just sort of... underwhelming.
The 2010s had yielded a modest 1.6x speed up in CPU, or 6% per year.
GPUs had done a bit better with maybe 12% per year, and together they
eked out 17% compute gains per year over six years.
In my youth of the 1990s, we typically saw speed ups of 40% per year
on real-world tasks.  In five years, your machine would be 5x as fast,
not 2x as fast.  Taking a turn down the
[memory lane](https://books.google.com/books?id=elneMPYGaagC&lpg=PP1&pg=PA48-IA8#v=onepage&q&f=false)
of computers I once owned you can taste the time of infinite speedups:

## Benchmark: [Matrix factorization](https://3dfmaps.com/CPU/cpu.htm)

Launch | Build | CPU     | benchmark seconds | speedup  | per year
-------|-------|---------|-------------------|----------|---------
1985   |  1991 | 386/25  | 437 seconds       |  1.0     |
1992   |  1994 | 486DX2/66 | 50 seconds      |  8.7x    |  1.36x
1994   |  1997 | Pentium 100 | 15.7 seconds  |  27.8x   |  1.45x
1998   |  1999 | Pentium II 450 | 1.68 seconds | 260x   |  1.48x


## Benchmark: [Doom Frames per Second](https://www.complang.tuwien.ac.at/misc/doombench.html)
Launch | Build | CPU     | benchmark fps | speedup  | per year
-------|-------|---------|---------------|----------|---------
1992   |  1994 | 486DX2/66 | 35fps       |  1.0     |
1994   |  1997 | Pentium 100 | 73fps     |  2x      |  1.43x
1998   |  1999 | Pentium II 450 | 129fps |  3.68x   |  1.24x

# Coda

After futzing about with my old rig, I discovered the cpu cooler
had gotten knocked off and just needed some new thermal paste.  So
what is one to do with an expensive, noisy, yet only twice as fast
new machine?  Am I going to be doing enough full-time ML research
to justify the cost of the upgrade (which still needs some Noctuas
to replace the shitty OEM fans)?  Maybe my old rig is good enough
for me and this beast can go to some intense gamer or
GPT enthusiast.

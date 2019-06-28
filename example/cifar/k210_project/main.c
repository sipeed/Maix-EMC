/* Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include "kpu.h"
#include <platform.h>
#include <printf.h>
#include <string.h>
#include <stdlib.h>
#include <sysctl.h>
#include "uarths.h"
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#include "w25qxx.h"


#define CLASS10 1

#define PLL0_OUTPUT_FREQ 1000000000UL
#define PLL1_OUTPUT_FREQ 400000000UL
#define PLL2_OUTPUT_FREQ 45158400UL

#define MODEL_ADDRESS 0xa00000

volatile uint32_t g_ai_done_flag;

extern const unsigned char gImage_image[] __attribute__((aligned(128)));
kpu_model_context_t task1;



INCBIN(model, "cifar10_output.kmodel");
//INCBIN(model, "cifar10_dense.kmodel");
//INCBIN(model, "cifar10_pool2.kmodel");
//INCBIN(model, "cifar10_batch2.kmodel");
//INCBIN(model, "cifar10.kmodel");
//#define KMODEL_SIZE (250*1024)
//uint8_t model_data[KMODEL_SIZE] __attribute__((aligned(128)));
 
#define _D printf("###%d\r\n",__LINE__);




static void ai_done(void* userdata)
{
    g_ai_done_flag = 1;
    
    float *features;
    size_t count;

    kpu_get_output(&task1, 0, (uint8_t **)&features, &count);
    count /= sizeof(float);
    printf("total %ld feature\r\n", count);
    if(count>1024) count=1024;
    size_t i;
    for (i = 0; i < count; i++)
    {
        if (i % 64 == 0)
            printf("\n");
        printf("%f, ", features[i]);
    }

    printf("\n");
}

int main()
{
    /* Set CPU and dvp clk */
    //sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    //sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    plic_init();
    sysctl_enable_irq();
    printf("start load kmodel!\r\n");
	/*w25qxx_init(3, 0);
    w25qxx_enable_quad_mode();
    w25qxx_read_data(MODEL_ADDRESS, model_data, KMODEL_SIZE, W25QXX_QUAD_FAST);
	*/
    if (kpu_load_kmodel(&task1, model_data) != 0)
    {
        printf("Cannot load kmodel.\n");
        exit(-1);
    }
    printf("start ai cal!\r\n");
    int j;
    for (j = 0; j < 1; j++)
    {
        g_ai_done_flag = 0;

        if (kpu_run_kmodel(&task1, gImage_image, 5, ai_done, NULL) != 0)
        {
            printf("Cannot run kmodel.\n");
            exit(-1);
        }
		while (!g_ai_done_flag);
    }
    printf("ai cal done!\r\n");

    while (1)
        ;    
}
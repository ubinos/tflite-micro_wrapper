#include <ubinos.h>

#if (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__STM3221GEVAL)
#if defined(STM32CUBEF2__DTTY_STM32_UART_ENABLE)

#include "main.h"
#include "stm32f2xx_it.h"

/**
 * @brief  This function handles DTTY_STM32_UART interrupt request.
 * @param  None
 * @retval None
 */
void DTTY_STM32_UART_IRQHandler(void)
{
    HAL_UART_IRQHandler(&DTTY_STM32_UART_HANDLE);
}

#endif /* defined(STM32CUBEF2__DTTY_STM32_UART_ENABLE) */
#endif /* (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__STM3221GEVAL) */

